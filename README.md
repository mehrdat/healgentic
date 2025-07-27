# Medical Diagnosis AI System

A sophisticated multi-agent medical diagnosis system using LangGraph and Together AI.

## ⚠️ IMPORTANT DISCLAIMER
This system is for **EXPERIMENTAL and EDUCATIONAL purposes ONLY**. 
Never use for actual medical diagnosis without proper medical supervision.

## Project Structure

```
medical_diagnosis_ai/
├── src/
│   ├── agents/              # Medical diagnosis agents
│   │   ├── __init__.py
│   │   ├── symptom_analysis.py
│   │   ├── differential_diagnosis.py
│   │   ├── question_generation.py
│   │   ├── diagnosis_finalization.py
│   │   └── medication_recommendation.py
│   ├── knowledge/           # Knowledge base and document processing
│   │   ├── __init__.py
│   │   ├── knowledge_base.py
│   │   └── embeddings.py
│   ├── llm/                 # LLM integrations
│   │   ├── __init__.py
│   │   └── together_client.py
│   ├── workflow/            # LangGraph workflow
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── graph.py
│   └── main.py             # Main application entry point
├── web_apps/               # Web interfaces
│   ├── streamlit_app.py
│   └── gradio_app.py
├── data/                   # Data storage
│   └── medical_textbooks/  # PDF/EPUB medical textbooks
├── tests/                  # Unit tests
├── config/                 # Configuration files
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
├── setup.py               # Installation setup
├── .env                   # Environment variables
└── README.md              # This file
```

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
streamlit run web_apps/streamlit_app.py

# Gradio
python web_apps/gradio_app.py
```

## Configuration

Set your Together AI API key in `.env`:
```
TOGETHER_API_KEY=your_api_key_here
TOGETHER_MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
```

## Medical Textbooks

Place your medical textbooks (PDF/EPUB) in the `data/medical_textbooks/` directory.
Supported books include:
- Harrison's Principles of Internal Medicine
- Goldman-Cecil Medicine
- Bates' Guide to Physical Examination
- And more...

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## License

Educational use only. Not for commercial medical diagnosis.
