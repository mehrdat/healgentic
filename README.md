# Medical Diagnosis AI System

A simple multi-agent medical diagnosis system using LangGraph and Together AI.

## ⚠️ IMPORTANT DISCLAIMER
This system is for **EXPERIMENTAL and EDUCATIONAL purposes ONLY**. 
Never use for actual medical diagnosis without proper medical supervision.


## Features

- **Multi-Agent Architecture**: Specialized agents for different diagnosis phases
- **Together AI Integration**: Advanced LLM capabilities with Qwen model
- **Medical Knowledge Base**: Vector database of medical textbooks
- **Web Interfaces**: Both Streamlit and Gradio implementations
- **Comprehensive Workflow**: From symptom analysis to medication recommendations

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`
4. Initialize knowledge base: `python src/main.py --init-kb`

## Usage

### Command Line
```bash
python src/main.py --symptoms "headache, fever, nausea"
```

### Web Interface
```bash
# Streamlit
#still working on it.
```

## Configuration

Set your AI API key in `.env`:
```


```

## Medical Textbooks

Place your medical textbooks (PDF/EPUB) in the `data/medical_textbooks/` directory.
Supported books include:
- Harrison's Principles of Internal Medicine
- Goldman-Cecil Medicine
- Bates' Guide to Physical Examination
- And more...

## License

Educational use only. Not for commercial medical diagnosis.
