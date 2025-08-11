import gradio as gr
import sys
from pathlib import Path
import json
import hashlib

# Add the src directory to the path
sys.path.insert(0, str((Path(__file__).parent / "src").resolve()))

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
        self.chat_history = []  # list[{"role","content"}]
        self.patient_info = {}

    def initialize_system(self):
        try:
            if self.system is None:
                self.system = MedicalDiagnosisSystem()
            return "✅ Medical Diagnosis System initialized successfully!"
        except Exception as e:
            return f"❌ Error initializing system: {str(e)}"

    def generate_contextual_response(self, diagnosis_state, current_question):
        differential = diagnosis_state.get("differential_diagnosis", {})
        hypotheses = differential.get("hypotheses", [])
        qt = (current_question.get("text") or "").lower()
        responses = []
        if hypotheses:
            top = hypotheses[0].get("condition", "")
            if any(w in qt for w in ["pain", "severity", "scale", "rate"]):
                responses = [
                    f"I'm thinking this could be {top}. The severity will help me confirm this.",
                    f"Based on your symptoms, {top} is possible. Let me check the intensity.",
                    f"The pain level will help distinguish between {top} and other conditions.",
                    f"I need to understand how severe this is to narrow down from {top}.",
                ]
            elif any(w in qt for w in ["when", "time", "duration", "long", "started"]):
                responses = [
                    f"The timing could help confirm if this is {top}.",
                    f"I'm considering {top} - the timeline will be revealing.",
                    f"When symptoms started matters for distinguishing {top} from other causes.",
                    f"The duration pattern could point to {top} or rule it out.",
                ]
            elif any(w in qt for w in ["where", "location", "area", "side", "part"]):
                responses = [
                    f"The exact location will help me determine if this fits {top}.",
                    f"I'm leaning toward {top}, but need to confirm the affected area.",
                    f"Location is key - {top} has a typical pattern.",
                    f"Where you feel this could confirm my suspicion of {top}.",
                ]
            else:
                responses = [
                    f"This will help me determine if {top} is the right diagnosis.",
                    f"I'm investigating whether this fits the {top} pattern.",
                    f"This question will help narrow down from {top} to the exact cause.",
                    f"I need this detail to confirm my thinking about {top}.",
                ]
            if len(hypotheses) > 1:
                second = hypotheses[1].get("condition", "")
                responses.extend([
                    f"I'm deciding between {top} and {second}.",
                    f"This will help me choose between {top} or {second}.",
                ])
        else:
            responses = [
                "This detail will help me understand what's happening.",
                "I'm gathering information to identify the underlying cause.",
            ]
        qid = current_question.get("id", "default")
        idx = int(hashlib.md5(qid.encode()).hexdigest(), 16) % len(responses)
        return responses[idx]

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

    def start_diagnosis(self, symptoms, messages):
        """Start the diagnosis process"""
        if not self.system:
            messages = messages or []
            messages.append({"role": "assistant", "content": "Please initialize the system first"})
            return gr.update(value=messages), "", gr.update(visible=False), ""

        if not symptoms.strip():
            messages = messages or []
            messages.append({"role": "assistant", "content": "Please describe your symptoms"})
            return gr.update(value=messages), "", gr.update(visible=False), ""

        if not self.patient_info:
            messages = messages or []
            messages.append({"role": "assistant", "content": "Please fill out patient information first"})
            return gr.update(value=messages), "", gr.update(visible=False), ""

        # Add user message
        messages = messages or []
        messages.append({"role": "user", "content": symptoms})

        try:
            # Start interactive diagnosis
            result = self.system.workflow.start_interactive_diagnosis(symptoms, self.patient_info)

            status = result.get("status")
            if status == "question_pending":
                self.current_question = result.get("question")
                self.diagnosis_state = result.get("state")
                self.diagnosis_started = True

                # Add assistant response
                messages.append({
                    "role": "assistant",
                    "content": (
                        "I've analyzed your symptoms. To provide an accurate diagnosis, "
                        "I need to ask you some specific questions."
                    ),
                })

                question_text = (self.current_question or {}).get("text", "")
                return gr.update(value=messages), "", self.create_question_interface(self.current_question), question_text

            elif status == "diagnosis_complete":
                diagnosis_text = self.format_diagnosis_results(result)
                messages.append({"role": "assistant", "content": diagnosis_text})

                # Reset state
                self.current_question = None
                self.diagnosis_state = None
                self.diagnosis_started = False

                return gr.update(value=messages), "", gr.update(visible=False), ""

        except Exception as e:
            error_msg = f"❌ Error starting diagnosis: {str(e)}"
            messages.append({"role": "assistant", "content": error_msg})
            return gr.update(value=messages), "", gr.update(visible=False), ""

        # Fallback
        return gr.update(value=messages), "", gr.update(visible=False), ""

    def create_question_interface(self, question):
        """Create the appropriate interface for the question type"""
        # Always use a Textbox-compatible update to avoid schema mismatches
        placeholder = "Type your answer here"
        if (question or {}).get("type") in ("select", "radio", "multiselect"):
            opts = (question or {}).get("options", [])
            if opts:
                placeholder = f"Options: {', '.join(map(str, opts))}"
        return gr.update(visible=True, label="Your answer", value="", placeholder=placeholder)

    def answer_question(self, answer, messages):
        """Process the answer to current question"""
        if not self.current_question or not self.system:
            return gr.update(), gr.update(visible=False), ""

        if answer is None or (isinstance(answer, str) and not answer.strip()):
            messages = messages or []
            messages.append({"role": "assistant", "content": "Please provide an answer before submitting."})
            return gr.update(value=messages), gr.update(visible=False), ""

        try:
            # Process the answer
            question_id = (self.current_question or {}).get("id", "")
            result = self.system.workflow.answer_question(question_id, answer, self.diagnosis_state)

            # Add answer to chat history
            question_text = (self.current_question or {}).get("text", "")
            messages = messages or []
            messages.append({"role": "user", "content": f"{question_text}: {answer}"})

            status = result.get("status")
            if status == "question_pending":
                # More questions needed
                self.current_question = result.get("question")
                self.diagnosis_state = result.get("state")

                # Generate contextual response
                contextual_message = self.generate_contextual_response(self.diagnosis_state, self.current_question)
                messages.append({"role": "assistant", "content": contextual_message})

                # Show next question
                question_text = (self.current_question or {}).get("text", "")
                return gr.update(value=messages), self.create_question_interface(self.current_question), question_text

            elif status == "diagnosis_complete":
                # Diagnosis complete
                self.current_question = None
                self.diagnosis_state = None
                self.diagnosis_started = False

                # Display comprehensive diagnosis results
                diagnosis_text = self.format_diagnosis_results(result)
                messages.append({"role": "assistant", "content": diagnosis_text})

                return gr.update(value=messages), gr.update(visible=False), ""

        except Exception as e:
            error_msg = f"❌ Error processing answer: {str(e)}"
            messages = messages or []
            messages.append({"role": "assistant", "content": error_msg})
            return gr.update(value=messages), gr.update(visible=False), ""

        # Fallback
        return gr.update(value=messages), gr.update(visible=False), ""

    def skip_question(self, messages):
        """Skip the current question"""
        if not self.current_question or not self.system:
            return gr.update(), gr.update(visible=False), ""

        try:
            question_id = (self.current_question or {}).get("id", "")
            result = self.system.workflow.answer_question(question_id, "Not provided", self.diagnosis_state)

            # Add skip to chat history
            question_text = (self.current_question or {}).get("text", "")
            messages = messages or []
            messages.append({"role": "user", "content": f"{question_text}: Skipped"})

            status = result.get("status")
            if status == "question_pending":
                self.current_question = result.get("question")
                self.diagnosis_state = result.get("state")

                contextual_message = self.generate_contextual_response(self.diagnosis_state, self.current_question)
                messages.append({"role": "assistant", "content": f"That's okay. {contextual_message}"})

                question_text = (self.current_question or {}).get("text", "")
                return gr.update(value=messages), self.create_question_interface(self.current_question), question_text

            elif status == "diagnosis_complete":
                self.current_question = None
                self.diagnosis_state = None
                self.diagnosis_started = False

                diagnosis_text = self.format_diagnosis_results(result)
                messages.append({"role": "assistant", "content": diagnosis_text})

                return gr.update(value=messages), gr.update(visible=False), ""

        except Exception as e:
            error_msg = f"❌ Error skipping question: {str(e)}"
            messages = messages or []
            messages.append({"role": "assistant", "content": error_msg})
            return gr.update(value=messages), gr.update(visible=False), ""

        # Fallback
        return gr.update(value=messages), gr.update(visible=False), ""

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
                    show_label=True,
                    type="messages"
                )
                
                # Question area
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
                
                status_btn = gr.Button("Get System Status")
                system_status = gr.Textbox(label="System Information", lines=10, interactive=False)
        
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
            outputs=[chatbot, symptoms_input, question_input, question_text]
        )
        
        submit_btn.click(
            app.answer_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, question_text]
        )
        
        skip_btn.click(
            app.skip_question,
            inputs=[chatbot],
            outputs=[chatbot, question_input, question_text]
        )
        
        kb_btn.click(
            app.initialize_knowledge_base,
            outputs=kb_status
        )
        
        status_btn.click(
            app.get_system_status,
            outputs=system_status
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
