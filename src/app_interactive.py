import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import MedicalDiagnosisSystem

st.set_page_config(
    page_title="Interactive Medical Diagnosis AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the system
@st.cache_resource
def initialize_system():
    """Initialize the medical diagnosis system"""
    return MedicalDiagnosisSystem()

def render_question_widget(question):
    """Render the appropriate widget based on question type"""
    question_type = question.get("type", "text")
    question_text = question.get("text", question.get("question", ""))
    question_id = question.get("id", question_text)
    
    st.write(f"**{question_text}**")
    
    answer = None
    
    if question_type == "slider":
        min_val = question.get("min", 0)
        max_val = question.get("max", 10)
        default_val = question.get("default", min_val)
        answer = st.slider(
            "Select value:",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            key=f"slider_{question_id}"
        )
        
    elif question_type == "select":
        options = question.get("options", ["Yes", "No"])
        answer = st.selectbox(
            "Choose one:",
            options,
            key=f"select_{question_id}"
        )
        
    elif question_type == "multiselect":
        options = question.get("options", [])
        answer = st.multiselect(
            "Choose all that apply:",
            options,
            key=f"multiselect_{question_id}"
        )
        
    elif question_type == "number":
        min_val = question.get("min", 0)
        max_val = question.get("max", 100)
        answer = st.number_input(
            "Enter number:",
            min_value=min_val,
            max_value=max_val,
            key=f"number_{question_id}"
        )
        
    elif question_type == "date":
        answer = st.date_input(
            "Select date:",
            key=f"date_{question_id}"
        )
        
    elif question_type == "radio":
        options = question.get("options", ["Yes", "No"])
        answer = st.radio(
            "Select one:",
            options,
            key=f"radio_{question_id}"
        )
        
    else:  # text input
        answer = st.text_input(
            "Enter your answer:",
            key=f"text_{question_id}"
        )
    
    return answer, question_id

def main():
    st.title("🏥 Interactive Medical Diagnosis AI System")
    st.markdown("---")
    
    # Initialize system
    if 'system' not in st.session_state:
        with st.spinner("Initializing Medical Diagnosis System..."):
            st.session_state.system = initialize_system()
    
    # Initialize session states
    if 'messages' not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', 'content': 'Hello! I\'m here to help with your medical concerns. Please fill out your information first.'}]
    
    if 'diagnosis_state' not in st.session_state:
        st.session_state.diagnosis_state = None
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    
    if 'diagnosis_started' not in st.session_state:
        st.session_state.diagnosis_started = False

    # Sidebar
    with st.sidebar:
        st.header("📋 System Information")
        
        if st.button("🔄 Initialize Knowledge Base"):
            with st.spinner("Initializing knowledge base..."):
                chunks = st.session_state.system.initialize_knowledge_base()
                st.success(f"Knowledge base initialized with {chunks} chunks")
        
        if st.button("📊 System Status"):
            status = st.session_state.system.get_system_status()
            st.json(status)
    
    # Patient information (mandatory before chat)
    st.subheader("👤 Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=150, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Not specified"])
    with col2:
        medical_history = st.text_area("Medical History", height=100)
        medications = st.text_area("Current Medications", height=100)

    patient_info = {
        "age": age,
        "gender": gender,
        "medical_history": medical_history,
        "medications": medications
    }

    # Chat Section
    st.subheader("💬 Medical Consultation")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Handle current question if one is pending
    if st.session_state.current_question:
        with st.chat_message("assistant"):
            st.write("Please answer the following question:")
            
            answer, question_id = render_question_widget(st.session_state.current_question)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Answer", type="primary"):
                    if answer is not None and str(answer).strip():
                        # Process the answer
                        print(f"🔍 Processing answer: {answer} for question: {question_id}")
                        result = st.session_state.system.workflow.answer_question(
                            question_id, answer, st.session_state.diagnosis_state
                        )
                        
                        print(f"📊 Result status: {result.get('status', 'unknown')}")
                        
                        # Add answer to chat history
                        st.session_state.messages.append({
                            "role": "user", 
                            "content": f"**{st.session_state.current_question.get('text', '')}**: {answer}"
                        })
                        
                        if result["status"] == "question_pending":
                            # More questions needed
                            print("➡️ More questions needed")
                            st.session_state.current_question = result["question"]
                            st.session_state.diagnosis_state = result["state"]
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Thank you! I need to ask you another question."
                            })
                        elif result["status"] == "diagnosis_complete":
                            # Diagnosis complete
                            print("🎉 Diagnosis complete! Clearing question state...")
                            st.session_state.current_question = None
                            st.session_state.diagnosis_state = None
                            st.session_state.diagnosis_started = False  # Reset for new diagnosis
                            
                            # Display diagnosis results
                            diagnosis_text = "## 🎯 Diagnosis Complete!\n\n"
                            
                            final_diagnosis = result.get("final_diagnosis", {})
                            if final_diagnosis:
                                diagnosis_text += f"**Primary Diagnosis:** {final_diagnosis.get('primary_diagnosis', 'Unknown')}\n\n"
                                diagnosis_text += f"**Confidence:** {result.get('confidence_score', 0):.1%}\n\n"
                                
                                if final_diagnosis.get('reasoning'):
                                    diagnosis_text += f"**Reasoning:** {final_diagnosis['reasoning']}\n\n"
                            
                            medications = result.get("medications", {})
                            if medications:
                                diagnosis_text += "### 💊 Treatment Recommendations:\n"
                                if medications.get("medications"):
                                    diagnosis_text += "**Medications:**\n"
                                    for med in medications["medications"]:
                                        diagnosis_text += f"- {med}\n"
                                    diagnosis_text += "\n"
                                
                                if medications.get("lifestyle_recommendations"):
                                    diagnosis_text += "**Lifestyle Recommendations:**\n"
                                    for rec in medications["lifestyle_recommendations"]:
                                        diagnosis_text += f"- {rec}\n"
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": diagnosis_text
                            })
                        else:
                            # Handle unexpected status
                            print(f"⚠️ Unexpected result status: {result.get('status', 'unknown')}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"⚠️ Unexpected result: {result.get('status', 'unknown')}. Please try again."
                            })
                            st.session_state.current_question = None
                            st.session_state.diagnosis_state = None
                            st.session_state.diagnosis_started = False
                            
                        print("🔄 Triggering rerun...")
                        st.rerun()
                    else:
                        st.warning("Please provide an answer before submitting.")
            
            with col2:
                if st.button("Skip Question"):
                    # Skip this question
                    print(f"⏭️ Skipping question: {question_id}")
                    result = st.session_state.system.workflow.answer_question(
                        question_id, "Not provided", st.session_state.diagnosis_state
                    )
                    
                    print(f"📊 Skip result status: {result.get('status', 'unknown')}")
                    
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"**{st.session_state.current_question.get('text', '')}**: Skipped"
                    })
                    
                    if result["status"] == "question_pending":
                        st.session_state.current_question = result["question"]
                        st.session_state.diagnosis_state = result["state"]
                    elif result["status"] == "diagnosis_complete":
                        print("🎉 Diagnosis complete after skip!")
                        st.session_state.current_question = None
                        st.session_state.diagnosis_state = None
                        st.session_state.diagnosis_started = False
                        
                        # Add diagnosis results to chat
                        diagnosis_text = "## 🎯 Diagnosis Complete!\n\n"
                        final_diagnosis = result.get("final_diagnosis", {})
                        if final_diagnosis:
                            diagnosis_text += f"**Primary Diagnosis:** {final_diagnosis.get('primary_diagnosis', 'Unknown')}\n\n"
                            diagnosis_text += f"**Confidence:** {result.get('confidence_score', 0):.1%}\n\n"
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": diagnosis_text
                        })
                    else:
                        st.session_state.current_question = None
                    
                    st.rerun()
    
    # Chat input for initial symptoms
    if not st.session_state.current_question:
        if symptoms := st.chat_input("Describe your symptoms or ask a medical question..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": symptoms})
            
            if not st.session_state.diagnosis_started:
                # Start interactive diagnosis
                with st.spinner("Analyzing your symptoms..."):
                    result = st.session_state.system.workflow.start_interactive_diagnosis(
                        symptoms, patient_info
                    )
                
                if result["status"] == "question_pending":
                    st.session_state.current_question = result["question"]
                    st.session_state.diagnosis_state = result["state"]
                    st.session_state.diagnosis_started = True
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I've analyzed your symptoms. To provide an accurate diagnosis, I need to ask you some specific questions."
                    })
                elif result["status"] == "diagnosis_complete":
                    # Unlikely but handle it
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Based on your symptoms, here's my assessment..."
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I'm sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                    })
            else:
                # Continue conversation
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I understand. Please continue answering the questions so I can help you better."
                })
            
            st.rerun()

if __name__ == "__main__":
    main()
