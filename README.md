---
title: Medical Diagnosis AI
emoji: 🏥
colorFrom: blue
colorTo: green
pinned: false
sdk: docker
app_port: 7860
app_file: app_gradio.py
pinned: false
license: mit

## 🏥 Medical Diagnosis AI System (Docker)

This Space runs the app inside a Docker container. The default command launches the Gradio app (`app_gradio.py`) on port 7860.

## 🤖 Smart LLM Selection

This app automatically detects its environment and chooses the appropriate AI model:

- **🌐 Hugging Face Spaces**: Uses free Hugging Face models (no API key needed!)
- **💻 Local/Laptop**: Uses Google Gemini (requires GOOGLE_API_KEY)

No configuration needed - it just works! 🎉

## Features

- **Interactive Diagnosis**: AI asks targeted questions based on symptoms
- **Dynamic Question Types**: Sliders, dropdowns, text inputs, and more
- **Contextual Responses**: AI provides reasoning for each question
- **Treatment Recommendations**: Filtered and categorized suggestions
- **Patient Information**: Comprehensive medical history collection
- **Dual Environment Support**: Works both locally and on Hugging Face

## How to Use

1. **Fill Patient Information**: Enter age, gender, medical history
2. **Initialize System**: Click "Initialize System" button
3. **Describe Symptoms**: Tell the AI what's bothering you
4. **Answer Questions**: Respond to AI's targeted questions
5. **Get Diagnosis**: Receive detailed diagnosis and treatment recommendations

## Important Disclaimer

⚠️ **This AI system is for educational purposes only.**

- Always consult with qualified healthcare providers
- Not for emergency medical situations
- Should not replace professional medical advice
- Contact emergency services for urgent medical needs

## Technical Details

- **Frontend**: Gradio interface (served on port 7860)
- **AI Orchestration**: Uses LangChain for workflow management
- **LLM Models**:
  - Hugging Face Spaces: Local Transformers (free)
  - Local: Google Gemini (requires API key)
- **Knowledge**: Vector database for medical information storage
- **Smart Deployment**: Automatic environment detection

---

### Run locally with Docker (optional)

```bash
docker build -t healgentic .
docker run -it -p 7860:7860 healgentic
```

If you prefer FastAPI health probe, you can mount `health:app` via uvicorn by changing the Docker CMD.

**Remember: Always see a real doctor for health problems!**
