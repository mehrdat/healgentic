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

def generate_contextual_response(diagnosis_state, current_question):
    """Generate a contextual response based on current diagnosis state and question"""
    
    # Get current differential diagnosis
    differential = diagnosis_state.get("differential_diagnosis", {})
    hypotheses = differential.get("hypotheses", [])
    
    # Get the current question to understand what we're investigating
    question_text = current_question.get("text", "").lower()
    question_reasoning = current_question.get("reasoning", "")
    
    # Generate contextual responses based on top hypotheses and question type
    if hypotheses:
        top_condition = hypotheses[0].get("condition", "")
        
        # Pain/severity related questions
        if any(word in question_text for word in ["pain", "severity", "scale", "rate"]):
            responses = [
                f"I'm thinking this could be {top_condition}. The severity will help me confirm this.",
                f"Based on your symptoms, {top_condition} is possible. Let me check the intensity.",
                f"The pain level will help distinguish between {top_condition} and other conditions.",
                f"I need to understand how severe this is to narrow down from {top_condition}.",
            ]
        
        # Timing/duration questions
        elif any(word in question_text for word in ["when", "time", "duration", "long", "started"]):
            responses = [
                f"The timing could help confirm if this is {top_condition}.",
                f"I'm considering {top_condition} - the timeline will be revealing.",
                f"When symptoms started matters for distinguishing {top_condition} from other causes.",
                f"The duration pattern could point to {top_condition} or rule it out.",
            ]
        
        # Location/area questions
        elif any(word in question_text for word in ["where", "location", "area", "side", "part"]):
            responses = [
                f"The exact location will help me determine if this fits {top_condition}.",
                f"I'm leaning toward {top_condition}, but need to confirm the affected area.",
                f"Location is key - {top_condition} has a typical pattern.",
                f"Where you feel this could confirm my suspicion of {top_condition}.",
            ]
        
        # Associated symptoms
        elif any(word in question_text for word in ["other", "additional", "along", "associated", "also"]):
            responses = [
                f"I'm checking for other signs that would support {top_condition}.",
                f"These additional symptoms could confirm {top_condition}.",
                f"Looking for the complete picture - {top_condition} often has related symptoms.",
                f"Other symptoms will help me distinguish {top_condition} from similar conditions.",
            ]
        
        # Triggers/causes
        elif any(word in question_text for word in ["trigger", "cause", "worse", "better", "aggravate"]):
            responses = [
                f"Understanding triggers will help confirm if this is {top_condition}.",
                f"What makes it worse could point to {top_condition} specifically.",
                f"I'm exploring if the pattern matches {top_condition}.",
                f"Triggers are diagnostic clues for {top_condition}.",
            ]
        
        # Medical history
        elif any(word in question_text for word in ["history", "before", "previous", "past", "family"]):
            responses = [
                f"Your medical background could explain why {top_condition} developed.",
                f"Past history might connect to {top_condition} or suggest alternatives.",
                f"I'm checking if your history supports the {top_condition} diagnosis.",
                f"Previous conditions could be linked to {top_condition}.",
            ]
        
        # Default responses
        else:
            responses = [
                f"This will help me determine if {top_condition} is the right diagnosis.",
                f"I'm investigating whether this fits the {top_condition} pattern.",
                f"This question will help narrow down from {top_condition} to the exact cause.",
                f"I need this detail to confirm my thinking about {top_condition}.",
                f"This could be the key to confirming {top_condition}.",
            ]
        
        # Add some variety with multiple conditions if available
        if len(hypotheses) > 1:
            second_condition = hypotheses[1].get("condition", "")
            additional_responses = [
                f"I'm deciding between {top_condition} and {second_condition}.",
                f"This will help me choose between {top_condition} or {second_condition}.",
                f"Could be {top_condition}, but {second_condition} is also possible.",
                f"I'm narrowing down from {top_condition} and {second_condition}.",
            ]
            responses.extend(additional_responses)
    
    else:
        # Fallback responses when no hypotheses available
        responses = [
            "This detail will help me understand what's happening.",
            "I'm gathering information to identify the underlying cause.",
            "This will help me narrow down the possibilities.",
            "I need this to build a clearer picture.",
            "This information is important for the diagnosis.",
        ]
    
    # Use hash of question ID to get consistent but varied responses
    import hashlib
    question_id = current_question.get("id", "default")
    hash_val = int(hashlib.md5(question_id.encode()).hexdigest(), 16)
    selected_response = responses[hash_val % len(responses)]
    
    return selected_response

def filter_treatment_suggestions(suggestions):
    """Filter out repetitive, obvious, or overly generic treatment suggestions"""
    
    # Phrases to filter out (case insensitive)
    filter_phrases = [
        "consult a doctor",
        "see a doctor", 
        "speak with your doctor",
        "discuss with your doctor",
        "seek medical attention",
        "practice relaxation techniques",
        "deep breathing exercises",
        "meditation",
        "stress management",
        "get plenty of rest",
        "maintain a healthy lifestyle",
        "stay hydrated",
        "eat a balanced diet",
        "for diagnosis and treatment"
    ]
    
    filtered_suggestions = []
    for suggestion in suggestions:
        suggestion_text = suggestion.get("suggestion", "").lower()
        
        # Skip if contains any filter phrases
        should_filter = any(phrase in suggestion_text for phrase in filter_phrases)
        
        # Also filter very short or generic suggestions
        if len(suggestion_text.strip()) < 20:
            should_filter = True
            
        if not should_filter:
            filtered_suggestions.append(suggestion)
    
    return filtered_suggestions

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
        
        # Medical Disclaimer Section
        st.markdown("---")
        st.header("⚠️ Important Notice")
        st.markdown("""
        **Medical Disclaimer:**
        
        This AI system is for educational and informational purposes only. It should **never** replace professional medical advice, diagnosis, or treatment.
        
        **Always consult** with a qualified healthcare provider for:
        - Medical diagnosis
        - Treatment decisions  
        - Medication changes
        - Emergency situations
        
        If you have a medical emergency, contact emergency services immediately.
        """)
        
        st.markdown("---")
    
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
            #st.write("Please answer the following question:")
            
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
                            
                            # Generate contextual response instead of repetitive message
                            contextual_message = generate_contextual_response(
                                st.session_state.diagnosis_state, 
                                result["question"]
                            )
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": contextual_message
                            })
                        elif result["status"] == "diagnosis_complete":
                            # Diagnosis complete
                            print("🎉 Diagnosis complete! Clearing question state...")
                            st.session_state.current_question = None
                            st.session_state.diagnosis_state = None
                            st.session_state.diagnosis_started = False  # Reset for new diagnosis
                            
                            # Display comprehensive diagnosis results
                            diagnosis_text = "## 🎯 Diagnosis Complete!\n\n"
                            
                            # Primary Diagnosis
                            final_diagnosis = result.get("final_diagnosis", {})
                            if final_diagnosis:
                                diagnosis_text += f"**🏥 Primary Diagnosis:** {final_diagnosis.get('primary_diagnosis', 'Unknown')}\n\n"
                                diagnosis_text += f"**📊 Confidence:** {result.get('confidence_score', 0):.1%}\n\n"
                                
                                if final_diagnosis.get('final_summary'):
                                    diagnosis_text += f"**📋 Summary:** {final_diagnosis['final_summary']}\n\n"
                                
                                if final_diagnosis.get('next_steps'):
                                    diagnosis_text += "**🔍 Next Steps:**\n\n"
                                    for step in final_diagnosis['next_steps']:
                                        diagnosis_text += f"  • {step}\n\n"
                                    diagnosis_text += "\n"
                            
                            # Treatment Plan - FIXED DATA ACCESS
                            medications = result.get("medications", {})
                            print(f"🔍 Debug - medications data: {medications}")
                            
                            if medications:
                                diagnosis_text += "## 💊 Treatment Recommendations:\n\n"
                                
                                suggestions = medications.get("suggestions", [])
                                print(f"🔍 Debug - suggestions: {suggestions}")
                                
                                if suggestions:
                                    # Filter out repetitive suggestions
                                    filtered_suggestions = filter_treatment_suggestions(suggestions)
                                    print(f"🔍 Debug - filtered suggestions: {filtered_suggestions}")
                                    
                                    if filtered_suggestions:
                                        suggestions_by_category = {}
                                        for suggestion in filtered_suggestions:
                                            category = suggestion.get("category", "General")
                                            if category not in suggestions_by_category:
                                                suggestions_by_category[category] = []
                                            suggestions_by_category[category].append(suggestion.get("suggestion", ""))
                                        
                                        category_icons = {
                                            "Lifestyle": "🏃",
                                            "Home Care": "🏠", 
                                            "When to See a Doctor": "👩‍⚕️",
                                            "Monitoring": "📊",
                                            "General": "ℹ️"
                                        }
                                        
                                        for category, category_suggestions in suggestions_by_category.items():
                                            icon = category_icons.get(category, "•")
                                            diagnosis_text += f"**{icon} {category}:**\n\n"
                                            for suggestion in category_suggestions:
                                                if suggestion:
                                                    diagnosis_text += f"  • {suggestion}\n\n"
                                            diagnosis_text += "\n"
                                    else:
                                        diagnosis_text += "No specific actionable recommendations available.\n\n"
                                else:
                                    diagnosis_text += "No specific actionable recommendations available.\n\n"
                                
                                # Remove important note and disclaimer from main output
                            
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
                        
                        # Add contextual message for skipped questions too
                        contextual_message = generate_contextual_response(
                            st.session_state.diagnosis_state, 
                            result["question"]
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"That's okay. {contextual_message}"
                        })
                    elif result["status"] == "diagnosis_complete":
                        print("🎉 Diagnosis complete after skip!")
                        st.session_state.current_question = None
                        st.session_state.diagnosis_state = None
                        st.session_state.diagnosis_started = False
                        
                        # Add comprehensive diagnosis results to chat
                        diagnosis_text = "## 🎯 Diagnosis Complete!\n\n"
                        
                        final_diagnosis = result.get("final_diagnosis", {})
                        if final_diagnosis:
                            diagnosis_text += f"**🏥 Primary Diagnosis:** {final_diagnosis.get('primary_diagnosis', 'Unknown')}\n\n"
                            diagnosis_text += f"**📊 Confidence:** {result.get('confidence_score', 0):.1%}\n\n"
                            
                            if final_diagnosis.get('final_summary'):
                                diagnosis_text += f"**📋 Summary:** {final_diagnosis['final_summary']}\n\n"
                            
                            if final_diagnosis.get('next_steps'):
                                diagnosis_text += "**🔍 Next Steps:**\n\n"
                                for step in final_diagnosis['next_steps']:
                                    diagnosis_text += f"  • {step}\n\n"
                                diagnosis_text += "\n"
                        
                        # Treatment recommendations - FIXED
                        medications = result.get("medications", {})
                        if medications:
                            diagnosis_text += "## 💊 Treatment Recommendations:\n\n"
                            
                            suggestions = medications.get("suggestions", [])
                            if suggestions:
                                # Filter out repetitive suggestions
                                filtered_suggestions = filter_treatment_suggestions(suggestions)
                                
                                if filtered_suggestions:
                                    suggestions_by_category = {}
                                    for suggestion in filtered_suggestions:
                                        category = suggestion.get("category", "General")
                                        if category not in suggestions_by_category:
                                            suggestions_by_category[category] = []
                                        suggestions_by_category[category].append(suggestion.get("suggestion", ""))
                                    
                                    category_icons = {
                                        "Lifestyle": "🏃",
                                        "Home Care": "🏠", 
                                        "When to See a Doctor": "👩‍⚕️",
                                        "Monitoring": "📊",
                                        "General": "ℹ️"
                                    }
                                    
                                    for category, category_suggestions in suggestions_by_category.items():
                                        icon = category_icons.get(category, "•")
                                        diagnosis_text += f"**{icon} {category}:**\n\n"
                                        for suggestion in category_suggestions:
                                            if suggestion:
                                                diagnosis_text += f"  • {suggestion}\n\n"
                                        diagnosis_text += "\n"
                            
                            if medications.get("important_note"):
                                diagnosis_text += f"**⚠️ Important Note:** {medications['important_note']}\n\n"
                        
                        # Remove disclaimer from main output
                        
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
                with st.spinner("Analyzing your prompt..."):
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
                    print("🎉 Diagnosis complete after skip!")
                    st.session_state.current_question = None
                    st.session_state.diagnosis_state = None
                    st.session_state.diagnosis_started = False
                    
                    # Use the same comprehensive display logic
                    diagnosis_text = "## 🎯 Diagnosis Complete!\n\n"
                    
                    final_diagnosis = result.get("final_diagnosis", {})
                    if final_diagnosis:
                        diagnosis_text += f"**🏥 Primary Diagnosis:** {final_diagnosis.get('primary_diagnosis', 'Unknown')}\n\n"
                        diagnosis_text += f"**📊 Confidence:** {result.get('confidence_score', 0):.1%}\n\n"
                        
                        if final_diagnosis.get('final_summary'):
                            diagnosis_text += f"**📋 Summary:** {final_diagnosis['final_summary']}\n\n"
                    
                    # Treatment recommendations
                    medications = result.get("medications", {})
                    if medications and medications.get("suggestions"):
                        diagnosis_text += "## 💊 Treatment Recommendations:\n\n"
                        
                        # Filter suggestions
                        filtered_suggestions = filter_treatment_suggestions(medications["suggestions"])
                        
                        if filtered_suggestions:
                            suggestions_by_category = {}
                            for suggestion in filtered_suggestions:
                                category = suggestion.get("category", "General")
                                if category not in suggestions_by_category:
                                    suggestions_by_category[category] = []
                                suggestions_by_category[category].append(suggestion["suggestion"])
                            
                            category_icons = {
                                "Lifestyle": "🏃",
                                "Home Care": "🏠", 
                                "When to See a Doctor": "👩‍⚕️",
                                "Monitoring": "📊",
                                "General": "ℹ️"
                            }
                            
                            for category, suggestions in suggestions_by_category.items():
                                icon = category_icons.get(category, "•")
                                diagnosis_text += f"**{icon} {category}:**\n\n"
                                for suggestion in suggestions:
                                    diagnosis_text += f"  • {suggestion}\n\n"
                                diagnosis_text += "\n"
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": diagnosis_text
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
