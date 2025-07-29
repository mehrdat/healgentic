"""
Warning suppression utilities
"""
import os
import warnings
import logging

def suppress_warnings():
    """Suppress various warning messages"""
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings
    
    # Suppress LangChain/LangSmith warnings
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    
    # Suppress general Python warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Suppress specific library warnings
    warnings.filterwarnings("ignore", module="transformers")
    warnings.filterwarnings("ignore", module="torch")
    warnings.filterwarnings("ignore", module="tensorflow")
    
    # Set logging levels for noisy libraries
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("langchain").setLevel(logging.ERROR)
    
    print("🔇 Warnings suppressed")

# Apply suppression immediately when module is imported
suppress_warnings()