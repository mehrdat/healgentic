"""
Streamlit Web Application for Medical Diagnosis AI System
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

from main import MedicalDiagnosisSystem

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the system
@st.cache_resource
def initialize_system():
    """Initialize the medical diagnosis system"""
    return MedicalDiagnosisSystem()

def main():
    st.title("🏥 Medical Diagnosis AI System")
    st.markdown("---")
    
    # Initialize system
    if 'system' not in st.session_state:
        with st.spinner("Initializing Medical Diagnosis System..."):
            st.session_state.system = initialize_system()
    
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
    
    # Main content
    st.header("🩺 Medical Diagnosis")
    
    # Input section
    symptoms = st.text_area(
        "Enter patient symptoms:",
        placeholder="Describe the symptoms in detail...",
        height=100
    )
    
    # Patient information (optional)
    with st.expander("Patient Information (Optional)"):
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
    
    # Diagnosis button
    if st.button("🔍 Run Diagnosis", type="primary"):
        if symptoms.strip():
            with st.spinner("Running medical diagnosis..."):
                try:
                    result = st.session_state.system.start_diagnosis(symptoms, patient_info)
                    
                    # Display results
                    st.header("📋 Diagnosis Results")
                    
                    # Final diagnosis
                    if "final_diagnosis" in result and result["final_diagnosis"]:
                        st.subheader("🎯 Primary Diagnosis")
                        diagnosis = result["final_diagnosis"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Condition", diagnosis.get("primary_diagnosis", "Unknown"))
                        with col2:
                            confidence = result.get("confidence_score", 0)
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        if "reasoning" in diagnosis:
                            st.write("**Reasoning:**", diagnosis["reasoning"])
                    
                    # Treatment plan
                    if "medications" in result and result["medications"]:
                        st.subheader("💊 Treatment Plan")
                        treatment = result["medications"]
                        
                        if "medications" in treatment:
                            st.write("**Recommended Medications:**")
                            for med in treatment["medications"]:
                                st.write(f"- {med}")
                        
                        if "lifestyle_recommendations" in treatment:
                            st.write("**Lifestyle Recommendations:**")
                            for rec in treatment["lifestyle_recommendations"]:
                                st.write(f"- {rec}")
                    
                    # Knowledge sources
                    if "knowledge_sources" in result and result["knowledge_sources"]:
                        st.subheader("📚 Knowledge Sources")
                        sources = list(set(result["knowledge_sources"]))  # Remove duplicates
                        for source in sources[:5]:  # Show top 5 sources
                            st.write(f"- {source}")
                    
                    # Raw result (for debugging)
                    with st.expander("🔧 Raw Diagnosis Data"):
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"Error during diagnosis: {str(e)}")
                    st.exception(e)
        else:
            st.warning("Please enter symptoms to run diagnosis.")

if __name__ == "__main__":
    main()
