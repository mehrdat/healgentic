# Simple linear workflow executor stub
class END:
    """Marker for end of workflow sequence"""
    pass

class StateGraph:
    """Executes added steps in sequence"""
    def __init__(self, state_type):
        self.steps = []

    def add_node(self, name, func):
        self.steps.append(func)

    def add_edge(self, *args):
        pass

    def set_entry_point(self, _):
        pass

    def compile(self):
        return self

    def invoke(self, state):
        for step in self.steps:
            state = step(state)
        return state

# Imports
from .state import MedicalDiagnosisState
from agents.initial_assessment_agent import get_initial_assessment_agent, InitialQuery
from agents.information_gathering_agent import get_information_gathering_agent
from agents.hypothesis_generation_agent import get_hypothesis_generation_agent
from agents.clarifying_question_agent import get_clarifying_question_agent
from agents.hypothesis_refinement_agent import get_hypothesis_refinement_agent
from agents.final_diagnosis_agent import get_final_diagnosis_agent
from agents.treatment_plan_agent import get_treatment_plan_agent

class MedicalDiagnosisWorkflow:
    """Linear workflow for medical diagnosis using a stub executor"""
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.setup_agents()
        self.setup_workflow()

    def setup_agents(self):
        print("🤖 Initializing medical diagnosis agents...")
        self.initial_assessment_agent = get_initial_assessment_agent()
        self.information_gathering_agent = get_information_gathering_agent()
        self.hypothesis_generation_agent = get_hypothesis_generation_agent()
        self.clarifying_question_agent = get_clarifying_question_agent()
        self.hypothesis_refinement_agent = get_hypothesis_refinement_agent()
        self.final_diagnosis_agent = get_final_diagnosis_agent()
        self.treatment_plan_agent = get_treatment_plan_agent()
        print("✅ All agents initialized")

    def setup_workflow(self):
        print("🔄 Setting up diagnosis workflow...")
        workflow = StateGraph(MedicalDiagnosisState)
        workflow.add_node("initial_assessment", self._initial_assessment_step)
        workflow.add_node("information_gathering", self._information_gathering_step)
        workflow.add_node("hypothesis_generation", self._hypothesis_generation_step)
        workflow.add_node("clarifying_questions", self._clarifying_questions_step)
        workflow.add_node("hypothesis_refinement", self._hypothesis_refinement_step)
        workflow.add_node("final_diagnosis", self._final_diagnosis_step)
        workflow.add_node("treatment_plan", self._treatment_plan_step)
        workflow.add_edge("initial_assessment", "information_gathering")
        workflow.add_edge("information_gathering", "hypothesis_generation")
        workflow.add_edge("hypothesis_generation", "clarifying_questions")
        workflow.add_edge("clarifying_questions", "hypothesis_refinement")
        workflow.add_edge("hypothesis_refinement", "final_diagnosis")
        workflow.add_edge("final_diagnosis", "treatment_plan")
        workflow.add_edge("treatment_plan", END)
        workflow.set_entry_point("initial_assessment")
        self.app = workflow.compile()
        print("✅ Workflow setup complete\n")

    def _initial_assessment_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        print("📝 Step 1: Initial Assessment")
        query = InitialQuery(text=state["user_symptoms"])
        assessment = self.initial_assessment_agent.invoke(query.dict())
        state["symptom_analysis"] = assessment.model_dump()
        state["current_step"] = "information_gathering"
        return state

    def _information_gathering_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        print("🔍 Step 2: Information Gathering")
        try:
            search_queries = self.information_gathering_agent.invoke(state["symptom_analysis"])
            retrieved_docs = []
            for query_obj in getattr(search_queries, 'queries', []):
                docs = self.knowledge_base.search_medical_knowledge(query_obj.query, k=3)
                if docs:
                    retrieved_docs.extend(docs)
            state["knowledge_sources"] = [doc.metadata.get("source_book", "Unknown") for doc in retrieved_docs] if retrieved_docs else []
            state["retrieved_knowledge"] = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
        except Exception as e:
            print(f"❌ Error searching knowledge base: {type(e).__name__}: {repr(e)}")
            import traceback
            traceback.print_exc()
            state["knowledge_sources"] = []
            state["retrieved_knowledge"] = ""
        state["current_step"] = "hypothesis_generation"
        return state

    def _hypothesis_generation_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        print("🧠 Step 3: Hypothesis Generation")
        differential = self.hypothesis_generation_agent.invoke({
            "assessment": state["symptom_analysis"],
            "retrieved_knowledge": state["retrieved_knowledge"]
        })
        state["differential_diagnosis"] = differential.model_dump()
        state["current_step"] = "clarifying_questions"
        return state

    def _clarifying_questions_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        print("❓ Step 4: Clarifying Questions")
        questions = self.clarifying_question_agent.invoke({
            "differential_diagnosis": state["differential_diagnosis"],
            "assessment": state["symptom_analysis"]
        })
        state["questions_asked"] = questions.model_dump()
        state["user_answers"] = self._simulate_user_answers(questions)
        state["current_step"] = "hypothesis_refinement"
        return state

    def _simulate_user_answers(self, questions) -> dict:
        answers = {}
        for i, q in enumerate(questions.questions):
            answers[q.question] = f"Simulated answer {i+1}"
        return answers

    def _hypothesis_refinement_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        print("� Step 5: Hypothesis Refinement")
        refined = self.hypothesis_refinement_agent.invoke({
            "differential_diagnosis": state["differential_diagnosis"],
            "user_answers": state["user_answers"]
        })
        state["differential_diagnosis"] = refined.dict()
        state["current_step"] = "final_diagnosis"
        return state

    def _final_diagnosis_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        print("🎯 Step 6: Final Diagnosis")
        final = self.final_diagnosis_agent.invoke({
            "refined_diagnosis": state["differential_diagnosis"]
        })
        state["final_diagnosis"] = final.dict()
        state["confidence_score"] = final.confidence_score
        state["current_step"] = "treatment_plan"
        return state

    def _treatment_plan_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        print("💊 Step 7: Treatment Plan")
        plan = self.treatment_plan_agent.invoke({
            "final_diagnosis": state["final_diagnosis"],
            "retrieved_knowledge": state["retrieved_knowledge"]
        })
        state["medications"] = plan.dict()
        state["current_step"] = "complete"
        return state

    def run_diagnosis(self, symptoms: str, patient_info: dict = None) -> dict:
        print(f"🩺 Starting medical diagnosis for: {symptoms[:50]}...")
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
            result = self.app.invoke(initial_state)
            print("✅ Diagnosis workflow completed")
            if result is None:
                return {"error": "Workflow returned None", "final_diagnosis": {"primary_diagnosis": "System Error", "confidence_score": 0.0}, "confidence_score": 0.0}
            return result
        except Exception as e:
            print(f"❌ Error in diagnosis workflow: {e}")
            return {"error": str(e), "final_diagnosis": {"primary_diagnosis": "System Error", "confidence_score": 0.0}, "confidence_score": 0.0}


    
