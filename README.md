---
title: Medical Diagnosis AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.37.2
app_file: app_gradio.py
pinned: false
license: mit
---

## 🏥 Medical Diagnosis AI System (Gradio)

An interactive AI system that helps with medical diagnosis through a question-and-answer interface.

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

- **Frontend**: Built with Gradio for interactive interface
- **AI Orchestration**: Uses LangChain for workflow management
- **LLM Models**:
  - Hugging Face Spaces: Local Transformers (free)
  - Local: Google Gemini (requires API key)
- **Knowledge**: Vector database for medical information storage
- **Smart Deployment**: Automatic environment detection

---

**Remember: Always see a real doctor for health problems!**

## 🚀 Deploy on Streamlit Cloud (keep big files off GitHub)

You can deploy this app from GitHub without pushing large vector-store files. The app will download its FAISS index at runtime from a Hugging Face repo.

1. Prepare your vector store on Hugging Face

  - Create a Dataset repo (recommended), e.g. `username/medical_kb_repo`
  - Upload `index.faiss` and `index.pkl` either in the repo root or under `medical_knowledge/`
  - For private repos, create an access token in your HF account

2. Set Streamlit Cloud environment variables

  - HF_VECTOR_STORE_REPO = `username/medical_kb_repo`
  - HF_REPO_TYPE = `dataset` (or `model`/`space` if that’s where you stored it)
  - HF_SUBFOLDER = `medical_knowledge` (only if you used that subfolder)
  - HUGGING_FACE_HUB_TOKEN = your token (only required for private repos)
  - LLM_PROVIDER = `hf` (default; uses local transformers)
  - HF_LOCAL_MODEL = `Qwen/Qwen2.5-0.5B-Instruct` (or another small CPU model, e.g., `google/flan-t5-base`)

3. Deploy the app

  - Create a new Streamlit Cloud app and point it to your GitHub repository
  - The app will auto-download the vector store on first run and cache it in `data/vector_store/medical_knowledge`

Notes

- The repository’s `.gitignore` excludes big files (FAISS, PKL); no Git LFS required
- You can manually sync from the sidebar if you prefer not to auto-sync on startup
- If you host files elsewhere (S3/GCS/public URL), the app can be adapted to download from there
