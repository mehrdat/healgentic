"""
LLM Configuration
-----------------
Preparing the LLM to use in the agent system.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from pathlib import Path

# Load environment variables from the .env file in the project root
# This line assumes your .env file is in the 'bio' directory
print("="*100)
dotenv_path = Path(__file__).resolve().parents[0].parent.parent / '.env'
load_dotenv(dotenv_path)


print(f"🔍 Loading environment variables from:\n {dotenv_path}")
def get_llm():
    """
    Initializes and returns the configured Language Model.
    
    Currently configured to use Google Gemini.
    
    Raises:
        ValueError: If the required API key is not found in the environment variables.
        
    Returns:
        An instance of a LangChain ChatModel.
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash") # Default model if not set
    
    
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please ensure it is set in your .env file."
        )

    # Initialize the ChatGoogleGenerativeAI model
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,  # Set to 0 for deterministic, factual responses
        max_tokens=900
    )
    
    return llm

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("Attempting to initialize the LLM...")
    try:
        llm_instance = get_llm()
        print("✅ LLM Initialized Successfully! \n")
        print(f"   - Provider: Google Gemini\n")
        print(f"   - Model: {llm_instance.model}\n")

        # Test invocation
        print("\nTesting LLM with a simple prompt...")
        response = llm_instance.invoke("Hello, how are you?")
        print(f"   - Response: {response.content}")
        
    except ValueError as e:
        print(f"❌ Error initializing LLM: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
