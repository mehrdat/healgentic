"""
LLM Configuration
-----------------
Prepares the LLM used in the agent system.
Uses a free local transformers model on Hugging Face Spaces, and Gemini locally.
Also wraps the LLM with a compatibility adapter that provides .with_structured_output
so older agent code will not crash.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Internal singleton cache for the LLM so we don't re-load weights every call
_LLM_SINGLETON = None  # type: ignore

# Disable LangSmith warnings and tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

# Load environment variables from the .env file in the project root
dotenv_path = Path(__file__).resolve().parents[0].parent.parent / ".env"
load_dotenv(dotenv_path)


def is_huggingface_space() -> bool:
    """Detect if we're running on Hugging Face Spaces."""
    return bool(os.getenv("SPACE_ID") or os.getenv("SPACE_AUTHOR_NAME"))


def _install_structured_output(llm):
    """Add with_structured_output(schema) to the LLM's class so instances remain Runnable.

    This avoids setting attributes on Pydantic BaseModel instances (like HuggingFacePipeline),
    which disallow new fields. Adding to the class is allowed and affects all instances safely.
    """
    try:
        from langchain_core.output_parsers import PydanticOutputParser
    except Exception:
        from langchain.output_parsers import PydanticOutputParser  # type: ignore
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
    except Exception:
        SystemMessage = None
        HumanMessage = None
    from langchain_core.runnables import RunnableLambda

    def _with_structured_output(self, schema):
        parser = PydanticOutputParser(pydantic_object=schema)
        fmt = parser.get_format_instructions()

        def _prepend_instructions(x):
            try:
                if SystemMessage and HumanMessage:
                    # Convert to messages with a leading system instruction
                    if isinstance(x, list):
                        return [SystemMessage(content="Output format (must strictly follow):\n" + fmt)] + x
                    elif isinstance(x, str):
                        return [SystemMessage(content="Output format (must strictly follow):\n" + fmt), HumanMessage(content=x)]
                    else:
                        return [SystemMessage(content="Output format (must strictly follow):\n" + fmt), HumanMessage(content=str(x))]
            except Exception:
                pass
            # Fallback: plain string prompt
            prefix = "Output format (must strictly follow):\n" + fmt + "\n\n"
            return prefix + (x if isinstance(x, str) else str(x))

        return RunnableLambda(_prepend_instructions) | self | parser

    cls = llm.__class__
    if not hasattr(cls, "with_structured_output"):
        try:
            setattr(cls, "with_structured_output", _with_structured_output)
        except Exception:
            # If class assignment somehow fails, just return llm without installing.
            pass
    return llm


def _wrap_llm_for_messages(llm):
    """Return a Runnable that coerces ChatMessages -> string for non-chat LLMs.

    - Accepts str or List[BaseMessage]. If a list, join content with role tags.
    - Preserves Runnable chaining semantics.
    - Mirrors with_structured_output on the wrapper so legacy calls still work.
    """
    try:
        from langchain_core.messages import BaseMessage
    except Exception:
        BaseMessage = None
    from langchain_core.runnables import RunnableLambda

    def preprocess(x):
        try:
            # If input is already a string, pass through
            if isinstance(x, str):
                return x
            # If messages list, convert to a simple prompt
            if BaseMessage is not None and isinstance(x, list) and all(hasattr(m, "content") for m in x):
                parts = []
                for m in x:
                    role = getattr(m, "type", getattr(m, "role", ""))
                    content = getattr(m, "content", "")
                    parts.append(f"[{role}] {content}")
                return "\n\n".join(parts)
            # If dict-like, just stringify
            return str(x)
        except Exception:
            return str(x)

    wrapper = RunnableLambda(preprocess) | llm

    # If underlying class has with_structured_output, reflect it on the wrapper too
    if hasattr(llm.__class__, "with_structured_output"):
        def _wrapper_with_structured(schema):
            return RunnableLambda(preprocess) | llm.with_structured_output(schema)
        try:
            setattr(wrapper, "with_structured_output", _wrapper_with_structured)  # type: ignore[attr-defined]
        except Exception:
            pass

    return wrapper


def reset_llm_cache():
    """Reset the cached LLM instance (next get_llm() call will rebuild)."""
    global _LLM_SINGLETON
    _LLM_SINGLETON = None


def get_llm():
    """Return a cached LLM instance.

    Caching avoids reinitializing / re-downloading the transformers pipeline on every
    agent call (a major source of latency on CPU Spaces). Set env LLM_DISABLE_CACHE=1
    to force rebuilding each call (for debugging/model switching).
    """
    global _LLM_SINGLETON
    if os.getenv("LLM_DISABLE_CACHE"):
        print("[LLM] Cache disabled via LLM_DISABLE_CACHE – rebuilding")
        reset_llm_cache()
    if _LLM_SINGLETON is not None:
        return _LLM_SINGLETON

    if is_huggingface_space():
        print("[LLM] Spaces detected: building local transformers model (cold start)")
        base = _install_structured_output(get_local_pipeline_llm())
        _LLM_SINGLETON = _wrap_llm_for_messages(base)
    else:
        print("[LLM] Local detected: attempting Gemini then HF fallback (cold start)")
        _LLM_SINGLETON = get_google_llm()
    return _LLM_SINGLETON


def get_google_llm():
    """Get Google Gemini LLM for local use (wrapped)."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment")
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0, max_tokens=900)
        print("[LLM] Gemini initialized:", model_name)
        llm = _install_structured_output(llm)
        return _wrap_llm_for_messages(llm)
    except Exception as e:
        print("[LLM] Gemini unavailable:", e, "— falling back to local HF pipeline")
    base = _install_structured_output(get_local_pipeline_llm())
    return _wrap_llm_for_messages(base)


def get_huggingface_llm():
    """Try Hugging Face Inference API (wrapped). Not used by default on Spaces."""
    try:
        try:
            from langchain_huggingface import HuggingFaceEndpoint
        except Exception:
            from langchain_community.llms import HuggingFaceEndpoint  # type: ignore
        model_name = "microsoft/DialoGPT-large"
        llm = HuggingFaceEndpoint(repo_id=model_name, temperature=0.1, max_new_tokens=512, repetition_penalty=1.1, return_full_text=False)
        print("[LLM] HF Endpoint initialized:", model_name)
        llm = _install_structured_output(llm)
        return _wrap_llm_for_messages(llm)
    except Exception as e:
        print("[LLM] HF Endpoint failed:", e, "— using local HF pipeline")
        base = _install_structured_output(get_local_pipeline_llm())
        return _wrap_llm_for_messages(base)


def get_local_pipeline_llm():
    """Use a local transformers pipeline (free)."""
    from transformers import pipeline
    try:
        from langchain_community.llms import HuggingFacePipeline
    except Exception:
        from langchain.llms import HuggingFacePipeline  # type: ignore

    model_name = os.getenv("HF_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    task = "text2text-generation" if any(k in model_name.lower() for k in ("t5", "flan")) else "text-generation"
    try:
        pipe = pipeline(task, model=model_name, max_new_tokens=384, temperature=0.1, do_sample=False, repetition_penalty=1.05)
    except Exception as e:
        fallback = "google/flan-t5-base"
        task = "text2text-generation"
        print(f"[LLM] Failed to load {model_name}: {e} — falling back to {fallback}")
        pipe = pipeline(task, model=fallback, max_new_tokens=256, temperature=0.1, do_sample=False, repetition_penalty=1.05)
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"[LLM] Local transformers ready: {model_name} ({task})")
    return llm


if __name__ == "__main__":
    print("Env:", "Spaces" if is_huggingface_space() else "Local")
    llm = get_llm()
    try:
        out = llm.invoke("Say hello in one short sentence.")
        print("Test:", getattr(out, "content", out))
    except Exception as e:
        print("Self-test failed:", e)
