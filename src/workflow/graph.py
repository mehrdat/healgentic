"""
LangGraph workflow definition for medical diagnosis using the new agent architecture
"""

from langgraph.graph import StateGraph, END
from .state import MedicalDiagnosisState
from agents.initial_assessment_agent import get_initial_assessment_agent, InitialQuery
from agents.information_gathering_agent import get_information_gathering_agent
from agents.hypothesis_generation_agent import get_hypothesis_generation_agent
from agents.clarifying_question_agent import get_clarifying_question_agent
from agents.hypothesis_refinement_agent import get_hypothesis_refinement_agent
from agents.final_diagnosis_agent import get_final_diagnosis_agent
from agents.treatment_plan_agent import get_treatment_plan_agent
from knowledge.knowledge_base import MedicalKnowledgeBase


class MedicalDiagnosisWorkflow:
    """LangGraph workflow for medical diagnosis using the new multi-agent system"""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.setup_agents()
        self.setup_workflow()
    
    def setup_agents(self):
        """Initialize all agents"""
        print("🤖 Initializing medical diagnosis agents...")
        
        # Initialize all the new agents
        self.initial_assessment_agent = get_initial_assessment_agent()
        self.information_gathering_agent = get_information_gathering_agent()
        self.hypothesis_generation_agent = get_hypothesis_generation_agent()
        self.clarifying_question_agent = get_clarifying_question_agent()
        self.hypothesis_refinement_agent = get_hypothesis_refinement_agent()
        self.final_diagnosis_agent = get_final_diagnosis_agent()
        self.treatment_plan_agent = get_treatment_plan_agent()
        
        print("✅ All agents initialized")
    
    def setup_workflow(self):
        """Setup LangGraph workflow"""
        print("🔄 Setting up diagnosis workflow...")
        
        workflow = StateGraph(MedicalDiagnosisState)
        
        # Add nodes for each step in the workflow
        workflow.add_node("initial_assessment", self._initial_assessment_step)
        workflow.add_node("information_gathering", self._information_gathering_step)
        workflow.add_node("hypothesis_generation", self._hypothesis_generation_step)
        workflow.add_node("clarifying_questions", self._clarifying_questions_step)
        workflow.add_node("hypothesis_refinement", self._hypothesis_refinement_step)
        workflow.add_node("final_diagnosis", self._final_diagnosis_step)
        workflow.add_node("treatment_plan", self._treatment_plan_step)
        
        # Define the workflow sequence
        workflow.add_edge("initial_assessment", "information_gathering")
        workflow.add_edge("information_gathering", "hypothesis_generation")
        workflow.add_edge("hypothesis_generation", "clarifying_questions")
        workflow.add_edge("clarifying_questions", "hypothesis_refinement")
        workflow.add_edge("hypothesis_refinement", "final_diagnosis")
        workflow.add_edge("final_diagnosis", "treatment_plan")
        workflow.add_edge("treatment_plan", END)
        
        # Set entry point
        workflow.set_entry_point("initial_assessment")
        
        # Compile the workflow
        self.app = workflow.compile()
        
        print("✅ Workflow setup complete")
    
    def _initial_assessment_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        """Step 1: Initial assessment of user symptoms"""
        print("📝 Step 1: Initial Assessment")
        
        query = InitialQuery(text=state["user_symptoms"])
        assessment = self.initial_assessment_agent.invoke(query.dict())
        
        state["symptom_analysis"] = assessment.dict()
        state["current_step"] = "information_gathering"
        
        return state
    
    def _information_gathering_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        """Step 2: Generate search queries and retrieve knowledge"""
        print("🔍 Step 2: Information Gathering")
        
        # Generate search queries
        search_queries = self.information_gathering_agent.invoke(state["symptom_analysis"])
        
        # Search the knowledge base with each query
        retrieved_docs = []
        for query_obj in search_queries.queries:
            docs = self.knowledge_base.similarity_search(query_obj.query, k=3)
            retrieved_docs.extend(docs)
        
        # Combine retrieved knowledge
        knowledge_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        state["knowledge_sources"] = [doc.metadata.get("source_book", "Unknown") for doc in retrieved_docs]
        state["retrieved_knowledge"] = knowledge_text
        state["current_step"] = "hypothesis_generation"
        
        return state
    
    def _hypothesis_generation_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        """Step 3: Generate differential diagnosis"""
        print("🧠 Step 3: Hypothesis Generation")
        
        differential = self.hypothesis_generation_agent.invoke({
            "assessment": state["symptom_analysis"],
            "retrieved_knowledge": state["retrieved_knowledge"]
        })
        
        state["differential_diagnosis"] = differential.dict()
        state["current_step"] = "clarifying_questions"
        
        return state
    
    def _clarifying_questions_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        """Step 4: Generate clarifying questions"""
        print("❓ Step 4: Clarifying Questions")
        
        questions = self.clarifying_question_agent.invoke({
            "differential_diagnosis": state["differential_diagnosis"],
            "assessment": state["symptom_analysis"]
        })
        
        state["questions_asked"] = questions.dict()
        state["current_step"] = "hypothesis_refinement"
        
        # For now, simulate user answers (in a real app, this would be interactive)
        # TODO: Make this interactive in the web interface
        state["user_answers"] = self._simulate_user_answers(questions)
        
        return state
    
    def _simulate_user_answers(self, questions) -> dict:
        """Simulate user answers for demonstration purposes"""
        # In a real application, this would be provided by the user
        answers = {}
        for i, q in enumerate(questions.questions):
            answers[q.question] = f"Simulated answer {i+1}"
        return answers
    
    def _hypothesis_refinement_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        """Step 5: Refine hypothesis based on answers"""
        print("🔄 Step 5: Hypothesis Refinement")
        
        refined_diagnosis = self.hypothesis_refinement_agent.invoke({
            "differential_diagnosis": state["differential_diagnosis"],
            "user_answers": state["user_answers"]
        })
        
        state["differential_diagnosis"] = refined_diagnosis.dict()
        state["current_step"] = "final_diagnosis"
        
        return state
    
    def _final_diagnosis_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        """Step 6: Final diagnosis"""
        print("🎯 Step 6: Final Diagnosis")
        
        final_diagnosis = self.final_diagnosis_agent.invoke({
            "refined_diagnosis": state["differential_diagnosis"]
        })
        
        state["final_diagnosis"] = final_diagnosis.dict()
        state["confidence_score"] = final_diagnosis.confidence_score
        state["current_step"] = "treatment_plan"
        
        return state
    
    def _treatment_plan_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        """Step 7: Treatment plan"""
        print("💊 Step 7: Treatment Plan")
        
        treatment_plan = self.treatment_plan_agent.invoke({
            "final_diagnosis": state["final_diagnosis"],
            "retrieved_knowledge": state["retrieved_knowledge"]
        })
        
        state["medications"] = treatment_plan.dict()
        state["current_step"] = "complete"
        
        return state
    
    def run_diagnosis(self, symptoms: str, patient_info: dict = None) -> dict:
        """Run the complete diagnosis workflow"""
        print(f"🩺 Starting medical diagnosis for: {symptoms[:50]}...")
        
        # Initialize state
        initial_state = MedicalDiagnosisState(
            user_symptoms=symptoms,
            patient_info=patient_info or {},
            symptom_analysis={},
            differential_diagnosis={},
            questions_asked={},
            user_answers={},
            final_diagnosis={},
            medications={},
            confidence_score=0.0,
            knowledge_sources=[],
            retrieved_knowledge="",
            current_step="initial_assessment"
        )
        
        try:
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            print("✅ Diagnosis workflow completed")
            print(f"📊 Final confidence: {result.get('confidence_score', 0):.2%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in diagnosis workflow: {e}")
            return {
                "error": str(e),
                "final_diagnosis": {"primary_diagnosis": "System Error", "confidence_score": 0.0},
                "confidence_score": 0.0
            }
    
    def get_workflow_status(self) -> dict:
        """Get workflow status and statistics"""
        return {
            "agents_initialized": hasattr(self, 'initial_assessment_agent'),
            "workflow_compiled": hasattr(self, 'app'),
            "knowledge_base_ready": not self.knowledge_base.demo_mode,
            "knowledge_base_stats": self.knowledge_base.get_statistics()
        }
