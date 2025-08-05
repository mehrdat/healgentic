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
        # Running on Hugging Face Spaces - use free models
        print("🌐 Detected Hugging Face Spaces environment")
        return get_huggingface_llm()
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
    """Get Hugging Face LLM for Spaces or fallback"""
    try:
        # Try different import paths for Hugging Face
        try:
            from langchain_huggingface import HuggingFaceEndpoint
        except ImportError:
            try:
                from langchain_community.llms import HuggingFaceEndpoint
            except ImportError:
                from langchain.llms import HuggingFacePipeline
                from transformers import pipeline
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
            print(f"⚠️ Hugging Face API error: {e}")
            # Fallback to GPT-2
            fallback_model = "gpt2"
            llm = HuggingFaceEndpoint(
                repo_id=fallback_model,
                temperature=0.1,
                max_new_tokens=256,
                repetition_penalty=1.1,
                return_full_text=False,
            )
            
            print(f"✅ Initialized fallback model: {fallback_model}")
            return llm
            
    except ImportError:
        print("⚠️ Hugging Face libraries not available, using local pipeline")
        return get_local_pipeline_llm()

def get_local_pipeline_llm():
    """Fallback to local transformers pipeline"""
    try:
        from transformers import pipeline
        from langchain.llms import HuggingFacePipeline
        
        pipe = pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ Initialized local transformers pipeline")
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
            print(f"   - Provider: Hugging Face (Free)\n")
        else:
            print(f"   - Provider: Google Gemini\n")
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
