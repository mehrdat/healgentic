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


def _with_structured_adapter(llm):
    """Return an adapter that implements with_structured_output(schema) using Pydantic."""
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

    class _Adapter:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def with_structured_output(self, schema):
            parser = PydanticOutputParser(pydantic_object=schema)
            fmt = parser.get_format_instructions()

            def _invoke(value):
                # Inject format instructions and call the inner LLM
                if isinstance(value, str):
                    if HumanMessage and SystemMessage:
                        msgs = [SystemMessage(content="Output format (must strictly follow):\n" + fmt), HumanMessage(content=value)]
                        out = self._inner.invoke(msgs)
                    else:
                        prompt = "Output format (must strictly follow):\n" + fmt + "\n\n" + value
                        out = self._inner.invoke(prompt)
                else:
                    # Fallback: stringify
                    prompt = "Output format (must strictly follow):\n" + fmt + "\n\n" + str(value)
                    out = self._inner.invoke(prompt)
                content = getattr(out, "content", out)
                return parser.parse(content)

            return RunnableLambda(_invoke)

    return _Adapter(llm)


def get_llm():
    """Initialize the LLM based on environment."""
    if is_huggingface_space():
        print("[LLM] Spaces detected: using local transformers model")
        return _with_structured_adapter(get_local_pipeline_llm())
    print("[LLM] Local detected: using Google Gemini (if available)")
    return get_google_llm()


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
        return _with_structured_adapter(llm)
    except Exception as e:
        print("[LLM] Gemini unavailable:", e, "— falling back to local HF pipeline")
        return _with_structured_adapter(get_local_pipeline_llm())


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
        return _with_structured_adapter(llm)
    except Exception as e:
        print("[LLM] HF Endpoint failed:", e, "— using local HF pipeline")
        return _with_structured_adapter(get_local_pipeline_llm())


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
