import os
import shutil
from pathlib import Path
import sys
import streamlit as st

# Ensure we can import from src/ as a package
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Medical Diagnosis AI", page_icon="🏥", layout="wide")


@st.cache_resource(show_spinner=False)
def get_system():
    from src.main import MedicalDiagnosisSystem
    return MedicalDiagnosisSystem()


def ensure_dirs():
    base = Path(__file__).parent
    (base / "data" / "vector_store").mkdir(parents=True, exist_ok=True)
    (base / "data" / "medical_textbooks").mkdir(parents=True, exist_ok=True)


def load_vector_store_from_hf(repo_id: str, subfolder=None, repo_type=None) -> str:
    """Download a prebuilt FAISS vector store from a Hugging Face repo into data/vector_store/medical_knowledge.

    Expects the repo (dataset or space or model) to contain the folder 'medical_knowledge' with index.faiss and index.pkl.
    """
    ensure_dirs()
    try:
        from huggingface_hub import snapshot_download, login
    except Exception as e:
        raise RuntimeError(f"huggingface_hub not available: {e}")

    # Token (optional for public repos)
    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        try:
            login(token=token)
        except Exception:
            pass

    # repo_type can be 'dataset', 'model', or 'space'; None lets the hub infer but is less reliable
    local_dir = snapshot_download(repo_id=repo_id, repo_type=repo_type, allow_patterns=None)
    src_root = Path(local_dir)
    if subfolder:
        src_root = src_root / subfolder

    src = src_root / "medical_knowledge"
    if not src.exists():
        # Try direct files in subfolder
        if (src_root / "index.faiss").exists() and (src_root / "index.pkl").exists():
            src = src_root
        else:
            raise FileNotFoundError("Could not find 'medical_knowledge' folder or FAISS files (index.faiss/index.pkl) in the specified repo/subfolder.")

    dst = Path(__file__).parent / "data" / "vector_store" / "medical_knowledge"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return str(dst)


st.title("🏥 Medical Diagnosis AI (Streamlit)")
st.markdown("This Space uses free Hugging Face models and a medical knowledge base.")

with st.sidebar:
    st.header("⚙️ Tools")

    # Reset session moved to sidebar
    if st.button("Reset Session", use_container_width=True):
        st.session_state.diag_state = None
        st.session_state.last_question_id = None
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    with st.expander("Advanced: Sync vector store from Hugging Face"):
        st.caption("Use this if you want to download a prebuilt FAISS index from your HF repo. If your database is already in /data/vector_store, you can ignore this.")
        repo = st.text_input("HF repo id", help="Format: namespace/name, e.g., username/medical_kb_repo")
        repo_type = st.selectbox("Repo type", ["dataset", "model", "space"], index=0, help="The type of Hugging Face repository. Choose 'dataset' if you stored the index in a dataset repo.")
        sub = st.text_input("Optional subfolder", help="Folder under the repo that contains 'medical_knowledge' or the index files.")
        if st.button("Download & Load", use_container_width=True, disabled=not repo.strip()):
            try:
                with st.spinner("Downloading vector store from Hugging Face..."):
                    path = load_vector_store_from_hf(repo_id=repo.strip(), subfolder=sub.strip() or None, repo_type=repo_type)
                st.success(f"Vector store synced to: {path}")
                st.info("Recreating system to pick up the new index...")
                get_system.clear()
                _sys = get_system()
                st.success("Vector store ready.")
            except Exception as e:
                st.error(f"Failed to load vector store: {e}")

    st.divider()
    st.subheader("📊 Status")
    if st.button("Show Status"):
        try:
            status = get_system().get_system_status()
            st.json(status)
        except Exception as e:
            st.error(f"Status error: {e}")

    # Inline controls are rendered next to the chat when needed


# Patient info will be collected through Q&A dynamically
patient = {}


# Diagnosis chat (full conversation view)
st.subheader("💬 Consultation")

# Session state init
if "diag_state" not in st.session_state:
    st.session_state.diag_state = None
if "last_question_id" not in st.session_state:
    st.session_state.last_question_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role: "user"|"assistant", content: str}

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))

# Helper: append a message
def append_message(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})

# Ensure the current question (if any) appears once in chat
def show_current_question_if_new(state_dict: dict):
    q = (state_dict or {}).get("question", {})
    qid = q.get("id")
    qtext = q.get("text")
    if qtext and qid and qid != st.session_state.last_question_id:
        append_message("assistant", qtext)
        st.session_state.last_question_id = qid

# Chat input (works for both first symptoms and subsequent answers)
user_input = st.chat_input("Type your symptoms to start, then answer questions here…")
if user_input:
    sysobj = get_system()
    # New conversation: treat input as symptoms
    if not st.session_state.diag_state:
        append_message("user", user_input)
        with st.spinner("Analyzing symptoms…"):
            res = sysobj.workflow.start_interactive_diagnosis(user_input, patient)
        st.session_state.diag_state = res
        if isinstance(res, dict) and res.get("status") == "question_pending":
            show_current_question_if_new(res)
        elif isinstance(res, dict) and res.get("status") == "diagnosis_complete":
            final = res.get("final_diagnosis", {})
            summary_lines = []
            if final.get("primary_diagnosis"):
                summary_lines.append(f"Primary diagnosis: {final.get('primary_diagnosis')}")
            if final.get("final_summary"):
                summary_lines.append(final.get("final_summary"))
            if summary_lines:
                append_message("assistant", "\n\n".join(summary_lines))
        st.rerun()
    else:
        # Ongoing Q&A: treat input as answer to the pending question
        state = st.session_state.diag_state
        status = (state or {}).get("status")
        append_message("user", user_input)
        if status == "question_pending":
            q = state.get("question", {})
            try:
                with st.spinner("Processing your answer…"):
                    next_state = sysobj.workflow.answer_question(q.get("id", ""), user_input, state.get("state"))
                st.session_state.diag_state = next_state
                if isinstance(next_state, dict) and next_state.get("status") == "question_pending":
                    show_current_question_if_new(next_state)
                elif isinstance(next_state, dict) and next_state.get("status") == "diagnosis_complete":
                    final = next_state.get("final_diagnosis", {})
                    summary_lines = []
                    if final.get("primary_diagnosis"):
                        summary_lines.append(f"Primary diagnosis: {final.get('primary_diagnosis')}")
                    if final.get("final_summary"):
                        summary_lines.append(final.get("final_summary"))
                    meds_section = next_state.get("medications", {}).get("suggestions", [])
                    meds_lines = [m.get("suggestion") for m in meds_section[:5] if m.get("suggestion")]
                    if meds_lines:
                        summary_lines.append("\n**Treatment suggestions:**\n- " + "\n- ".join(meds_lines))
                    if summary_lines:
                        append_message("assistant", "\n\n".join(summary_lines))
                st.rerun()
            except Exception as e:
                append_message("assistant", f"Error processing answer: {e}")
                st.rerun()
        else:
            # If conversation is completed, start a new one
            append_message("assistant", "Session complete. Click Reset Session to start over, or type new symptoms to begin a new session.")

# Controls
colA, colB = st.columns([1, 1])
with colA:
    st.caption("Use the chat to converse with the AI. It will ask follow-up questions.")
with colB:
    st.empty()

# Inline structured controls for the current question (context-aware UI)
state_now = st.session_state.get("diag_state")
if isinstance(state_now, dict) and state_now.get("status") == "question_pending":
    q = state_now.get("question", {})
    qid = q.get("id", "q")
    qtext = q.get("text", "")
    qtype = (q.get("type") or "text").lower()
    opts = q.get("options") or []
    qmin = q.get("min") if q.get("min") is not None else 0
    qmax = q.get("max") if q.get("max") is not None else 10
    qdef = q.get("default")

    # Only show UI controls when needed (non-text types)
    if qtype in {"slider", "radio", "select", "multiselect", "number", "date"}:
        st.markdown("---")
        st.caption("Answer using a control or type in the chat above.")
        with st.container(border=True):
            value = None
            if qtype == "slider":
                default_val = int(qdef) if isinstance(qdef, (int, float)) else int((qmin + qmax) / 2)
                value = st.slider(qtext or "Select a value", min_value=int(qmin), max_value=int(qmax), value=default_val, key=f"slider_{qid}")
            elif qtype == "radio":
                options = opts or ["Yes", "No"]
                value = st.radio(qtext or "Choose one", options, key=f"radio_{qid}")
            elif qtype == "select":
                options = opts or ["Option 1", "Option 2"]
                value = st.selectbox(qtext or "Select one", options, key=f"select_{qid}")
            elif qtype == "multiselect":
                options = opts or ["Option A", "Option B"]
                value = st.multiselect(qtext or "Select any", options, default=qdef or [], key=f"multi_{qid}")
            elif qtype == "number":
                default_val = int(qdef) if isinstance(qdef, (int, float)) else int(qmin)
                value = st.number_input(qtext or "Enter a number", min_value=int(qmin), max_value=int(qmax), value=default_val, key=f"num_{qid}")
            elif qtype == "date":
                value = st.date_input(qtext or "Pick a date", key=f"date_{qid}")

            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("Submit", key=f"submit_{qid}", use_container_width=True):
                    # Convert control value to string answer
                    if isinstance(value, list):
                        answer_text = ", ".join(map(str, value))
                    else:
                        answer_text = str(value)
                    if answer_text.strip():
                        append_message("user", answer_text)
                        try:
                            with st.spinner("Processing your answer…"):
                                next_state = get_system().workflow.answer_question(q.get("id", ""), answer_text, state_now.get("state"))
                            st.session_state.diag_state = next_state
                            if isinstance(next_state, dict) and next_state.get("status") == "question_pending":
                                show_current_question_if_new(next_state)
                            elif isinstance(next_state, dict) and next_state.get("status") == "diagnosis_complete":
                                final = next_state.get("final_diagnosis", {})
                                summary_lines = []
                                if final.get("primary_diagnosis"):
                                    summary_lines.append(f"Primary diagnosis: {final.get('primary_diagnosis')}")
                                if final.get("final_summary"):
                                    summary_lines.append(final.get("final_summary"))
                                meds_section = next_state.get("medications", {}).get("suggestions", [])
                                meds_lines = [m.get("suggestion") for m in meds_section[:5] if m.get("suggestion")]
                                if meds_lines:
                                    summary_lines.append("\n**Treatment suggestions:**\n- " + "\n- ".join(meds_lines))
                                if summary_lines:
                                    append_message("assistant", "\n\n".join(summary_lines))
                            st.rerun()
                        except Exception as e:
                            append_message("assistant", f"Error processing answer: {e}")
                            st.rerun()
            with col2:
                st.caption(q.get("reasoning") or "")

# Results panel (concise design) when diagnosis is complete
def _short_explainer(name: str) -> str:
    mapping = {
        "antacid": "Neutralizes stomach acid for quick relief.",
        "ppi": "Reduces acid production (once daily).",
        "h2": "Reduces acid, often at night.",
        "alginate": "Forms a protective raft to block reflux.",
        "ginger": "May soothe nausea and digestion.",
        "zinc": "Can irritate the esophagus; take with food.",
        "bed elevation": "Reduces nighttime reflux.",
        "weight": "Weight loss reduces abdominal pressure.",
    }
    key = name.lower()
    for k, v in mapping.items():
        if k in key:
            return v
    return "Commonly used option for symptom control."

def render_results(state_dict: dict):
    if not isinstance(state_dict, dict):
        return
    if state_dict.get("status") != "diagnosis_complete":
        return
    final = state_dict.get("final_diagnosis", {})
    meds_section = state_dict.get("medications", {}).get("suggestions", [])

    st.markdown("---")
    st.subheader("🎯 Result")
    c1, c2 = st.columns([2, 1])
    with c1:
        pdx = final.get("primary_diagnosis", "Unknown")
        st.markdown(f"### {pdx}")
        if final.get("final_summary"):
            st.markdown(final.get("final_summary"))
    with c2:
        st.metric("Confidence", f"{int(state_dict.get('confidence_score', 0.0)*100)}%")

    if meds_section:
        st.markdown("#### 💊 Options")
        for m in meds_section[:8]:
            s = (m or {}).get("suggestion")
            if not s:
                continue
            one_line = _short_explainer(s)
            with st.container(border=True):
                st.markdown(f"**{s}**  ")
                st.caption(one_line)
                details = (m or {}).get("details") or (m or {}).get("reasoning")
                with st.expander("Why this?"):
                    st.write(details or "Targets reflux/heartburn mechanisms.")

# Auto-render results if available
render_results(st.session_state.get("diag_state"))
