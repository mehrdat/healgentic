# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# This Dockerfile runs the Gradio app on port 7860 in a non-root user context.

FROM python:3.10-slim

# Create non-root user required by HF Dev Mode and to avoid permission issues
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies first (better caching)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository
COPY --chown=user . /app

# Switch to non-root user and set env
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface

# Default command launches the Streamlit app
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
