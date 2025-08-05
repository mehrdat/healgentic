import gradio as gr
import sys
from pathlib import Path
import json
import hashlib

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the medical diagnosis system
try:
    from main import MedicalDiagnosisSystem
    print("✅ Successfully imported MedicalDiagnosisSystem")
except ImportError as e:
    print(f"⚠️ Could not import MedicalDiagnosisSystem: {e}")
    print("🔄 Creating mock system for testing...")
    
    # Create a mock system for testing when dependencies are missing
    class MedicalDiagnosisSystem:
        def __init__(self):
            print("🧪 Mock Medical Diagnosis System initialized")
            self.workflow = self.MockWorkflow()
            
        def initialize_knowledge_base(self):
            return 0
            
        def get_system_status(self):
            return {"status": "mock system", "model": "test"}
            
        class MockWorkflow:
            def start_interactive_diagnosis(self, symptoms, patient_info):
                return {
                    "status": "diagnosis_complete",
                    "final_diagnosis": {
                        "primary_diagnosis": "Mock Diagnosis",
                        "final_summary": "This is a test diagnosis from the mock system."
                    },
                    "confidence_score": 0.85,
                    "medications": {
                        "suggestions": [
                            {"suggestion": "Rest and hydration", "category": "Home Care"},
                            {"suggestion": "Monitor symptoms", "category": "Monitoring"}
                        ]
                    }
                }
            
            def answer_question(self, question_id, answer, state):
                return {
                    "status": "diagnosis_complete", 
                    "final_diagnosis": {"primary_diagnosis": "Mock completion"}
                }

class GradioMedicalApp:
    def __init__(self):
        self.system = None
        self.diagnosis_state = None
        self.current_question = None
        self.diagnosis_started = False
        self.chat_history = []
        self.patient_info = {}
        
    def initialize_system(self):
        """Initialize the medical diagnosis system"""
        try:
            if self.system is None:
                self.system = MedicalDiagnosisSystem()
            return "✅ Medical Diagnosis System initialized successfully!"
        except Exception as e:
            return f"❌ Error initializing system: {str(e)}"
    
    def generate_contextual_response(self, diagnosis_state, current_question):
        """Generate a contextual response based on current diagnosis state and question"""
        
        # Get current differential diagnosis
        differential = diagnosis_state.get("differential_diagnosis", {})
        hypotheses = differential.get("hypotheses", [])
        
        # Get the current question to understand what we're investigating
        question_text = current_question.get("text", "").lower()
        
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
        question_id = current_question.get("id", "default")
        hash_val = int(hashlib.md5(question_id.encode()).hexdigest(), 16)
        selected_response = responses[hash_val % len(responses)]
        
        return selected_response

    def filter_treatment_suggestions(self, suggestions):
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

    def save_patient_info(self, age, gender, medical_history, medications):
        """Save patient information"""
        self.patient_info = {
            "age": age,
            "gender": gender,
            "medical_history": medical_history,
            "medications": medications
        }
        return f"✅ Patient information saved: {age} year old {gender}"

    def start_diagnosis(self, symptoms, history):
        """Start the diagnosis process"""
        if not self.system:
            return history + [["Error", "Please initialize the system first"]], "", gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        if not symptoms.strip():
            return history + [["Error", "Please describe your symptoms"]], "", gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        if not self.patient_info:
            return history + [["Error", "Please fill out patient information first"]], "", gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        # Add user message to history
        history.append([symptoms, None])
        
        try:
            # Start interactive diagnosis
            result = self.system.workflow.start_interactive_diagnosis(
                symptoms, self.patient_info
            )
            
            if result["status"] == "question_pending":
                self.current_question = result["question"]
                self.diagnosis_state = result["state"]
                self.diagnosis_started = True
                
                # Add assistant response
                history.append([None, "I've analyzed your symptoms. To provide an accurate diagnosis, I need to ask you some specific questions."])
                
                # Show the question interface
                question_text = self.current_question.get("text", "")
                
                return history, "", self.create_question_interface(self.current_question), question_text, gr.update(visible=True), gr.update(visible=True)
                
            elif result["status"] == "diagnosis_complete":
                # Format diagnosis results
                diagnosis_text = self.format_diagnosis_results(result)
                history.append([None, diagnosis_text])
                
                # Reset state
                self.current_question = None
                self.diagnosis_state = None
                self.diagnosis_started = False
                
                return history, "", gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        except Exception as e:
            error_msg = f"❌ Error starting diagnosis: {str(e)}"
            history.append([None, error_msg])
            return history, "", gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        return history, "", gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)

    def create_question_interface(self, question):
        """Create the appropriate interface for the question type"""
        question_type = question.get("type", "text")
        
        if question_type == "slider":
            min_val = question.get("min", 0)
            max_val = question.get("max", 10)
            default_val = question.get("default", min_val)
            return gr.update(visible=True, label=f"Scale (0-{max_val})", minimum=min_val, maximum=max_val, value=default_val)
        
        elif question_type in ["select", "radio"]:
            options = question.get("options", ["Yes", "No"])
            return gr.update(visible=True, choices=options, value=options[0] if options else None, label="Select one")
        
        elif question_type == "multiselect":
            options = question.get("options", [])
            return gr.update(visible=True, choices=options, value=[], label="Select all that apply")
        
        elif question_type == "number":
            min_val = question.get("min", 0)
            max_val = question.get("max", 100)
            return gr.update(visible=True, label="Enter number", minimum=min_val, maximum=max_val, value=min_val)
        
        else:  # text input
            return gr.update(visible=True, label="Enter your answer", value="")

    def answer_question(self, answer, history):
        """Process the answer to current question"""
        if not self.current_question or not self.system:
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            return history + [[None, "Please provide an answer before submitting."]], gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        try:
            # Process the answer
            question_id = self.current_question.get("id", "")
            result = self.system.workflow.answer_question(
                question_id, answer, self.diagnosis_state
            )
            
            # Add answer to chat history
            question_text = self.current_question.get('text', '')
            history.append([f"{question_text}: {answer}", None])
            
            if result["status"] == "question_pending":
                # More questions needed
                self.current_question = result["question"]
                self.diagnosis_state = result["state"]
                
                # Generate contextual response
                contextual_message = self.generate_contextual_response(
                    self.diagnosis_state, 
                    result["question"]
                )
                
                history.append([None, contextual_message])
                
                # Show next question
                question_text = self.current_question.get("text", "")
                return history, self.create_question_interface(self.current_question), question_text, gr.update(visible=True), gr.update(visible=True)
                
            elif result["status"] == "diagnosis_complete":
                # Diagnosis complete
                self.current_question = None
                self.diagnosis_state = None
                self.diagnosis_started = False
                
                # Display comprehensive diagnosis results
                diagnosis_text = self.format_diagnosis_results(result)
                history.append([None, diagnosis_text])
                
                return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        except Exception as e:
            error_msg = f"❌ Error processing answer: {str(e)}"
            history.append([None, error_msg])
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)

    def skip_question(self, history):
        """Skip the current question"""
        if not self.current_question or not self.system:
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        try:
            question_id = self.current_question.get("id", "")
            result = self.system.workflow.answer_question(
                question_id, "Not provided", self.diagnosis_state
            )
            
            # Add skip to chat history
            question_text = self.current_question.get('text', '')
            history.append([f"{question_text}: Skipped", None])
            
            if result["status"] == "question_pending":
                self.current_question = result["question"]
                self.diagnosis_state = result["state"]
                
                # Generate contextual response
                contextual_message = self.generate_contextual_response(
                    self.diagnosis_state, 
                    result["question"]
                )
                
                history.append([None, f"That's okay. {contextual_message}"])
                
                # Show next question
                question_text = self.current_question.get("text", "")
                return history, self.create_question_interface(self.current_question), question_text, gr.update(visible=True), gr.update(visible=True)
                
            elif result["status"] == "diagnosis_complete":
                # Diagnosis complete
                self.current_question = None
                self.diagnosis_state = None
                self.diagnosis_started = False
                
                # Display diagnosis results
                diagnosis_text = self.format_diagnosis_results(result)
                history.append([None, diagnosis_text])
                
                return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        except Exception as e:
            error_msg = f"❌ Error skipping question: {str(e)}"
            history.append([None, error_msg])
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
        
        return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)

    def format_diagnosis_results(self, result):
        """Format the diagnosis results for display"""
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
        
        # Treatment Plan
        medications = result.get("medications", {})
        if medications:
            diagnosis_text += "## 💊 Treatment Recommendations:\n\n"
            
            suggestions = medications.get("suggestions", [])
            if suggestions:
                # Filter out repetitive suggestions
                filtered_suggestions = self.filter_treatment_suggestions(suggestions)
                
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
        
        return diagnosis_text

    def initialize_knowledge_base(self):
        """Initialize the knowledge base"""
        if not self.system:
            return "❌ Please initialize the system first"
        
        try:
            chunks = self.system.initialize_knowledge_base()
            return f"✅ Knowledge base initialized with {chunks} chunks"
        except Exception as e:
            return f"❌ Error initializing knowledge base: {str(e)}"

    def get_system_status(self):
        """Get system status"""
        if not self.system:
            return "❌ System not initialized"
        
        try:
            status = self.system.get_system_status()
            return json.dumps(status, indent=2)
        except Exception as e:
            return f"❌ Error getting status: {str(e)}"

def create_app():
    """Create the Gradio interface"""
    app = GradioMedicalApp()
    
    with gr.Blocks(title="🏥 Medical Diagnosis AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 Interactive Medical Diagnosis AI System
        
        **⚠️ Medical Disclaimer:** This AI system is for educational purposes only. 
        Always consult with a qualified healthcare provider for medical advice, diagnosis, or treatment.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Medical Consultation",
                    height=400,
                    show_label=True
                )
                
                # Question interface (hidden by default)
                with gr.Group(visible=False) as question_group:
                    question_text = gr.Markdown("", visible=True)
                    
                    # Dynamic question input (will be updated based on question type)
                    question_input = gr.Textbox(
                        label="Your answer",
                        visible=False
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Submit Answer", variant="primary")
                        skip_btn = gr.Button("Skip Question", variant="secondary")
                
                # Symptoms input
                symptoms_input = gr.Textbox(
                    label="Describe your symptoms",
                    placeholder="Tell me what's bothering you...",
                    lines=3
                )
                
                start_btn = gr.Button("Start Diagnosis", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Patient Information
                gr.Markdown("### 👤 Patient Information")
                age_input = gr.Number(label="Age", value=30, minimum=0, maximum=150)
                gender_input = gr.Dropdown(
                    label="Gender",
                    choices=["Male", "Female", "Other", "Not specified"],
                    value="Not specified"
                )
                medical_history_input = gr.Textbox(
                    label="Medical History",
                    lines=3,
                    placeholder="Any chronic conditions, surgeries, etc."
                )
                medications_input = gr.Textbox(
                    label="Current Medications",
                    lines=3,
                    placeholder="List any medications you're taking"
                )
                
                save_info_btn = gr.Button("Save Patient Info", variant="secondary")
                patient_status = gr.Textbox(label="Status", interactive=False)
                
                # System Controls
                gr.Markdown("---")
                gr.Markdown("### 🔧 System Controls")
                
                init_btn = gr.Button("Initialize System", variant="primary")
                init_status = gr.Textbox(label="System Status", interactive=False)
                
                kb_btn = gr.Button("Initialize Knowledge Base")
                kb_status = gr.Textbox(label="Knowledge Base Status", interactive=False)
        
        # Event handlers
        init_btn.click(
            app.initialize_system,
            outputs=init_status
        )
        
        save_info_btn.click(
            app.save_patient_info,
            inputs=[age_input, gender_input, medical_history_input, medications_input],
            outputs=patient_status
        )
        
        start_btn.click(
            app.start_diagnosis,
            inputs=[symptoms_input, chatbot],
            outputs=[chatbot, symptoms_input, question_input, question_text, question_group, submit_btn]
        )
        
        submit_btn.click(
            app.answer_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, question_text, question_group, submit_btn]
        )
        
        skip_btn.click(
            app.skip_question,
            inputs=[chatbot],
            outputs=[chatbot, question_input, question_text, question_group, submit_btn]
        )
        
        kb_btn.click(
            app.initialize_knowledge_base,
            outputs=kb_status
        )
        
        # Add disclaimer at bottom
        gr.Markdown("""
        ---
        **⚠️ Important Notice:**
        - This AI is for educational purposes only
        - Always see a real doctor for health problems  
        - Don't use for emergencies
        - Contact emergency services for urgent medical needs
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch()
