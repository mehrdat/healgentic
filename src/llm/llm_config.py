"""
LLM Configuration
-----------------
Preparing the LLM to use in the agent system.
Supports both local (Google Gemini) and Hugging Face Spaces (free models).
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Disable LangSmith warnings and tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

# Load environment variables from the .env file in the project root
print("="*100)
dotenv_path = Path(__file__).resolve().parents[0].parent.parent / '.env'
load_dotenv(dotenv_path)

def is_huggingface_space():
    """Detect if we're running on Hugging Face Spaces"""
    return os.getenv("SPACE_ID") is not None or os.getenv("SPACE_AUTHOR_NAME") is not None

def get_llm():
    """
    Initializes and returns the configured Language Model.
    
    Automatically detects environment:
    - Hugging Face Spaces: Uses free Hugging Face models
    - Local/Other: Uses Google Gemini (requires API key)
    
    Returns:
        An instance of a LangChain ChatModel.
    """
    
    if is_huggingface_space():
        # Running on Hugging Face Spaces - use a free local transformers pipeline
        print("🌐 Detected Hugging Face Spaces environment (using local transformers model)")
        return get_local_pipeline_llm()
    else:
        # Running locally or elsewhere - use Google Gemini
        print("💻 Detected local environment")
        return get_google_llm()

def get_google_llm():
    """Get Google Gemini LLM for local use"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
        
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please ensure it is set in your .env file."
            )

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,  # Set to 0 for deterministic, factual responses
            max_tokens=900
        )
        
        print(f"✅ Initialized Google Gemini: {model_name}")
        return llm
        
    except ImportError:
        print("⚠️ Google Gemini not available, falling back to Hugging Face")
        return get_huggingface_llm()
    except Exception as e:
        print(f"⚠️ Error with Google Gemini: {e}, falling back to Hugging Face")
        return get_huggingface_llm()

def get_huggingface_llm():
    """Get Hugging Face LLM via Inference API (not used on Spaces by default)."""
    try:
        # Try different import paths for Hugging Face
        try:
            from langchain_huggingface import HuggingFaceEndpoint
        except ImportError:
            try:
                from langchain_community.llms import HuggingFaceEndpoint
            except ImportError:
                # Fall back to local pipeline interface
                return get_local_pipeline_llm()
        
        # Try Hugging Face Inference API first
        model_name = "microsoft/DialoGPT-large"
        
        try:
            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                temperature=0.1,
                max_new_tokens=512,
                repetition_penalty=1.1,
                return_full_text=False,
            )
            
            print(f"✅ Initialized Hugging Face model: {model_name}")
            return llm
            
        except Exception as e:
            print(f"⚠️ Hugging Face API error: {e}; falling back to local transformers pipeline")
            return get_local_pipeline_llm()
            
    except ImportError:
        print("⚠️ Hugging Face libraries not available, using local pipeline")
        return get_local_pipeline_llm()

def get_local_pipeline_llm():
    """Use a local transformers pipeline (free). Defaults to flan-t5-small for text2text-generation."""
    try:
        from transformers import pipeline
        try:
            # Prefer community import for newer LangChain versions
            from langchain_community.llms import HuggingFacePipeline
        except Exception:
            from langchain.llms import HuggingFacePipeline  # fallback

        # Allow override via env; default to a small, free model
        model_name = os.getenv("HF_LOCAL_MODEL", "google/flan-t5-small")

        # Choose task based on model family (simple heuristic)
        task = "text2text-generation" if any(k in model_name.lower() for k in ["t5", "flan"]) else "text-generation"

        pipe = pipeline(
            task,
            model=model_name,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.05,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"✅ Initialized local transformers pipeline: {model_name} ({task})")
        return llm
        
    except Exception as e:
        raise ValueError(f"Failed to initialize any LLM: {e}")

print(f"🔍 Environment: {'Hugging Face Spaces' if is_huggingface_space() else 'Local/Other'}")

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("Attempting to initialize the LLM...")
    try:
        llm_instance = get_llm()
        print("✅ LLM Initialized Successfully! \n")
        
        if is_huggingface_space():
            print("   - Provider: Hugging Face (Free)\n")
        else:
            print("   - Provider: Google Gemini\n")
            if hasattr(llm_instance, 'model'):
                print(f"   - Model: {llm_instance.model}\n")

        # Test invocation
        print("\nTesting LLM with a simple prompt...")
        response = llm_instance.invoke("Hello, how are you?")
        
        if hasattr(response, 'content'):
            print(f"   - Response: {response.content}")
        else:
            print(f"   - Response: {response}")
        
    except ValueError as e:
        print(f"❌ Error initializing LLM: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
