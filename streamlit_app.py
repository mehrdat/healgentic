"""
Streamlit Cloud entrypoint.

This shim simply imports app_streamlit so the app runs when Streamlit
auto-detects the default file name `streamlit_app.py`.
"""

# Executing the import runs the Streamlit application defined there.
import app_streamlit  # noqa: F401

