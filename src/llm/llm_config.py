"""
LLM Configuration
-----------------
This module provides a centralized function to configure and retrieve the 
Language Model (LLM) for the entire application.

It reads configuration details (API keys, model names) from the .env file.
"""
import os
from dotenv import load_dotenv
from langchain_together import ChatTogether

# Load environment variables from the .env file in the project root
# This line assumes your .env file is in the 'bio' directory
dotenv_path = os.path.join(os.path.dirname(__file__), '../../../.env')
load_dotenv(dotenv_path=dotenv_path)

def get_llm():
    """
    Initializes and returns the configured Language Model.
    
    Currently configured to use TogetherAI.
    
    Raises:
        ValueError: If the required API key is not found in the environment variables.
        
    Returns:
        An instance of a LangChain ChatModel.
    """
    api_key = os.getenv("TOGETHER_API_KEY")
    model_name = os.getenv("TOGETHER_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1") # Default model if not set

    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY not found in environment variables. "
            "Please ensure it is set in your .env file."
        )

    # Initialize the ChatTogether model
    llm = ChatTogether(
        model=model_name,
        together_api_key=api_key,
        temperature=0.0,  # Set to 0 for deterministic, factual responses
        max_tokens=2048
    )
    
    return llm

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("Attempting to initialize the LLM...")
    try:
        llm_instance = get_llm()
        print("✅ LLM Initialized Successfully!")
        print(f"   - Provider: TogetherAI")
        print(f"   - Model: {llm_instance.model}")
        
        # Test invocation
        print("\nTesting LLM with a simple prompt...")
        response = llm_instance.invoke("Hello, how are you?")
        print(f"   - Response: {response.content}")
        
    except ValueError as e:
        print(f"❌ Error initializing LLM: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
