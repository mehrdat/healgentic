# Google Gemini Setup Guide

## Overview

The Medical Diagnosis AI System has been updated to use Google Gemini (`models/gemini-2.5-flash-lite`) as the primary LLM provider.

## Setup Instructions

### 1. Get a Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

### 2. Update Environment Configuration

Add your Google API key to the `.env` file:

```bash
# Replace 'your_google_api_key_here' with your actual API key
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL=models/gemini-2.5-flash-lite
```

### 3. Test the Configuration

Run the LLM configuration test:

```bash
cd src/llm
python llm_config.py
```

You should see output like:
```
✅ LLM Initialized Successfully!
   - Provider: Google Gemini
   - Model: models/gemini-2.5-flash-lite
   - Response: [Gemini's response to "Hello, how are you?"]
```

### 4. Run the Medical Diagnosis System

```bash
# Test with sample symptoms
python src/main.py --symptoms "persistent headache for 3 days, nausea, sensitivity to light, mild fever"

# Initialize knowledge base (if not already done)
python src/main.py --init-kb
```

## Benefits of Google Gemini

- **Fast Response Times**: Gemini 2.5 Flash Lite is optimized for speed
- **Cost Effective**: Competitive pricing for API usage
- **High Quality**: Advanced reasoning capabilities for medical analysis
- **Reliable**: Google's robust infrastructure

## Troubleshooting

### API Key Issues
- Ensure the API key is correctly set in `.env`
- Verify the key has the necessary permissions
- Check that billing is enabled in your Google Cloud account

### Rate Limits
- Gemini has generous rate limits for most use cases
- Monitor usage in the Google AI Studio console

### Model Availability
- `models/gemini-2.5-flash-lite` is currently available in most regions
- If unavailable, try `models/gemini-1.5-flash` as an alternative

## Migration Notes

- The system previously used Together AI
- All agent configurations remain the same
- Only the underlying LLM provider has changed
- Existing knowledge base and vector store are compatible

## Support

For issues specific to Google Gemini integration:
- [Google AI Studio Documentation](https://ai.google.dev/docs)
- [Gemini API Reference](https://ai.google.dev/api)
