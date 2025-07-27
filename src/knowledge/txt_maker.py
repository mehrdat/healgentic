import streamlit as st
import os
import asyncio
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import shutil

# --- Core Conversion Functions (Refactored) ---

def get_output_path(input_path, output_folder):
    """Generates the output .txt path from an input file path."""
    filename = os.path.basename(input_path)
    name, _ = os.path.splitext(filename)
    return os.path.join(output_folder, f"{name}.txt")

def extract_text_from_epub(epub_path):
    """Extracts text from an EPUB file."""
    book = epub.read_epub(epub_path)
    full_text = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        soup = BeautifulSoup(content, 'html.parser')
        text = ' '.join(soup.get_text().strip().split())
        if text:
            full_text.append(text)
    return "\n\n".join(full_text)

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    This version does NOT perform OCR.
    """
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        
        # If the extracted text is empty, it's likely a scanned/image-only PDF.
        if not text.strip():
            filename = os.path.basename(pdf_path)
            return (
                f"NOTE: No text layer was found in '{filename}'.\n\n"
                "This file is likely a scanned document or contains only images. "
                "Text could not be extracted without OCR."
            )
        return text
    except Exception as e:
        filename = os.path.basename(pdf_path)
        return f"ERROR: Failed to process '{filename}' with PyMuPDF. The file might be corrupted or password-protected. Reason: {e}"

def process_file(file_path, output_folder):
    """
    Worker function to process a single file and save it as a .txt.
    """
    output_path = get_output_path(file_path, output_folder)
    if os.path.exists(output_path):
        return (file_path, "SKIPPED", "Already converted.")
    
    try:
        if file_path.lower().endswith('.epub'):
            text = extract_text_from_epub(file_path)
        elif file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            return (file_path, "SKIPPED", "Unsupported file type.")
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(text)
        return (file_path, "SUCCESS", None)
    except Exception as e:
        return (file_path, "FAILURE", str(e))

# --- Async Orchestrator (Unchanged) ---
async def run_conversion_async(files_to_process, output_folder, status_placeholder, progress_bar):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        tasks = [loop.run_in_executor(pool, process_file, f, output_folder) for f in files_to_process]
        total_files = len(tasks)
        completed_files = 0
        for future in asyncio.as_completed(tasks):
            try:
                file_path, status, reason = await future
                filename = os.path.basename(file_path)
                if status == "SUCCESS":
                    st.toast(f"✅ Converted: {filename}", icon="✅")
                elif status == "SKIPPED":
                    st.toast(f"⏩ Skipped: {filename}", icon="⏩")
                elif status == "FAILURE":
                    st.error(f"❌ Failed: {filename} - {reason}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
            
            completed_files += 1
            progress = completed_files / total_files
            status_placeholder.text(f"Processing... {completed_files}/{total_files} files complete.")
            progress_bar.progress(progress)

# --- Streamlit UI (Preserving your new features) ---
def main():
    st.set_page_config(page_title="Book Converter", layout="wide")
    st.title("📚 EPUB & PDF to TXT Converter")
    st.info("This app converts `.epub` and text-based `.pdf` files to `.txt` files. This version does not support OCR for scanned documents.")

    # Method selection
    st.header("1. Choose Input Method")
    method = st.radio(
        "How would you like to provide your books?",
        ["Upload Files", "Specify Folder Path"],
        horizontal=True
    )

    files_to_process = []
    
    if method == "Upload Files":
        st.header("2. Upload Your Books")
        uploaded_files = st.file_uploader(
            "Choose PDF or EPUB files",
            type=['pdf', 'epub'],
            accept_multiple_files=True,
            help="You can select multiple files at once"
        )
        
        if uploaded_files:
            temp_dir = st.session_state.get('temp_dir')
            if not temp_dir or not os.path.exists(temp_dir):
                temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
                os.makedirs(temp_dir, exist_ok=True)
                st.session_state['temp_dir'] = temp_dir
            
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                files_to_process.append(temp_file_path)
            
            st.success(f"✅ {len(files_to_process)} files ready for conversion!")
    
    else:  # Specify Folder Path
        st.header("2. Specify Folder Path")
        input_folder = st.text_input(
            "Enter the full path to your books folder:",
            placeholder="/path/to/your/books/folder",
            help="Enter the complete path to the folder containing your PDF and EPUB files"
        )
        
        if input_folder and os.path.isdir(input_folder):
            files_to_process = [
                os.path.join(input_folder, f) for f in os.listdir(input_folder)
                if f.lower().endswith(('.pdf', '.epub'))
            ]
            if files_to_process:
                st.success(f"✅ Found {len(files_to_process)} books in the folder!")
                with st.expander("📋 Files found"):
                    for file_path in files_to_process:
                        st.write(f"📖 {os.path.basename(file_path)}")
            else:
                st.warning("⚠️ No PDF or EPUB files found in the specified folder.")
        elif input_folder:
            st.error("❌ Invalid folder path. Please check the path and try again.")
    
    # Output folder selection
    st.header("3. Output Settings")
    output_method = st.radio(
        "Where should the converted files be saved?",
        ["Download converted files", "Save to specific folder"],
        horizontal=True
    )
    
    output_folder = None
    if output_method == "Save to specific folder":
        output_folder = st.text_input(
            "Enter output folder path:",
            placeholder="/path/to/output/folder",
            help="Enter the complete path where converted TXT files should be saved"
        )
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            st.info(f"📁 Output folder: {output_folder}")
    else:
        output_folder = os.path.join(os.getcwd(), 'converted_files')
        os.makedirs(output_folder, exist_ok=True)
        st.info("📥 Files will be available for download after conversion")
    
    # Start conversion
    st.header("4. Convert Files")
    
    if files_to_process and output_folder:
        if st.button("🚀 Start Conversion", type="primary", use_container_width=True):
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            status_placeholder.text("Starting conversion process...")
            
            asyncio.run(run_conversion_async(files_to_process, output_folder, status_placeholder, progress_bar))
            
            status_placeholder.success("🎉 All tasks complete!")
            st.balloons()
            
            if output_method == "Download converted files":
                st.header("📥 Download Converted Files")
                converted_files = [f for f in os.listdir(output_folder) if f.endswith('.txt')]
                
                if converted_files:
                    for txt_file in converted_files:
                        file_path = os.path.join(output_folder, txt_file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        
                        st.download_button(
                            label=f"📄 Download {txt_file}",
                            data=file_content,
                            file_name=txt_file,
                            mime='text/plain'
                        )
                else:
                    st.warning("No converted files found.")
            else:
                st.markdown(f"### ✅ Converted files saved to: `{output_folder}`")
    
    elif not files_to_process:
        st.info("👆 Please upload files or specify a valid folder path first.")
    elif not output_folder:
        st.info("👆 Please specify an output location.")
    
    # Cleanup section
    st.divider()
    if st.button("🧹 Clean Up Temporary Files"):
        temp_dir = st.session_state.get('temp_dir')
        output_temp_dir = os.path.join(os.getcwd(), 'converted_files')
        
        cleaned = False
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            if 'temp_dir' in st.session_state:
                del st.session_state['temp_dir']
            cleaned = True
        
        if os.path.exists(output_temp_dir):
            shutil.rmtree(output_temp_dir)
            cleaned = True
        
        if cleaned:
            st.success("🧹 Temporary files cleaned up!")
            st.rerun()
        else:
            st.info("No temporary files to clean up.")

if __name__ == "__main__":
    main()