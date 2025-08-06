
# Imports
from .state import MedicalDiagnosisState
from agents import (
    get_initial_assessment_agent,
    get_information_gathering_agent,
    get_hypothesis_generation_agent,
    get_clarifying_question_agent,
    get_hypothesis_refinement_agent,
    get_final_diagnosis_agent,
    get_treatment_plan_agent,
    InitialQuery
)
from utils.logging_utils import logger
from langgraph.graph import StateGraph, END
from langsmith import traceable
from typing import List




class MedicalDiagnosisWorkflow:
    """Linear workflow for medical diagnosis using a stub executor
    
    the methods:
    - setup_agents: Initializes all agents used in the workflow.
    - setup_workflow: Sets up the state graph for the diagnosis workflow.
    - _initial_assessment_step: Handles the initial assessment of user symptoms.
    - _information_gathering_step: Gathers information based on the initial assessment.
    - _hypothesis_generation_step: Generates differential diagnoses based on the assessment and knowledge base.
    - _clarifying_questions_step: Generates clarifying questions based on the differential diagnosis.
    - _hypothesis_refinement_step: Refines the differential diagnosis based on user answers.
    - _final_diagnosis_step: Generates the final diagnosis based on the refined differential diagnosis
    - _treatment_plan_step: Generates a treatment plan based on the final diagnosis and retrieved knowledge.
    - start_diagnosis: Starts the diagnosis workflow with the provided symptoms and patient information.
    - run_diagnosis: Runs the entire diagnosis workflow and returns the final results.
    
    
    
    """
    
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.setup_agents()
        self.setup_workflow()

    def setup_agents(self):
        logger.info("🤖 Initializing medical diagnosis agents...")
        self.initial_assessment_agent = get_initial_assessment_agent()
        self.information_gathering_agent = get_information_gathering_agent()
        self.hypothesis_generation_agent = get_hypothesis_generation_agent()
        self.clarifying_question_agent = get_clarifying_question_agent()
        self.hypothesis_refinement_agent = get_hypothesis_refinement_agent()
        self.final_diagnosis_agent = get_final_diagnosis_agent()
        self.treatment_plan_agent = get_treatment_plan_agent()
        logger.info("✅ All agents initialized")

    def setup_workflow(self):
        logger.info("🔄 Setting up diagnosis workflow...")
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
        
        workflow.add_conditional_edges(
            "clarifying_questions",
            self._should_continue_questioning,  # Decision function
            {
                "continue": "clarifying_questions",  # Loop back for more questions
                "finish": "hypothesis_refinement"     # Move to next step
            }
        )

        workflow.add_edge("hypothesis_refinement", "final_diagnosis")
        workflow.add_edge("final_diagnosis", "treatment_plan")
        workflow.add_edge("treatment_plan", END)
        
        workflow.set_entry_point("initial_assessment")
        self.app = workflow.compile()
        logger.info("✅ Workflow setup complete\n")


    @traceable(name="Step 1: Initial Assessment")
    def _initial_assessment_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        logger.info("Executing Step 1: Initial Assessment")
        
        query = InitialQuery(text=state["user_symptoms"])
        assessment = self.initial_assessment_agent.invoke(query.model_dump())
        
        state["symptom_analysis"] = assessment.model_dump()
        logger.debug(f"Symptom Analysis Output: {state['symptom_analysis']}")
        state["current_step"] = "information_gathering"
        return state

    @traceable(name="Step 2: Information Gathering")
    def _information_gathering_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        logger.info("Executing Step 2: Information Gathering")
        
        try:
            search_queries = self.information_gathering_agent.invoke(state["symptom_analysis"])
            logger.info(f"Generated Search Queries: {[q.query for q in getattr(search_queries, 'queries', [])]}")
            retrieved_docs = []
            for query_obj in getattr(search_queries, 'queries', []):
                docs = self.knowledge_base.search_medical_knowledge(query_obj.query, k=3)
                if docs:
                    retrieved_docs.extend(docs)
            
            state["knowledge_sources"] = [doc.metadata.get("source_book", "Unknown") for doc in retrieved_docs] if retrieved_docs else []
            state["retrieved_knowledge"] = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
            logger.info(f"Retrieved {len(retrieved_docs)} documents from the knowledge base.")
            logger.debug(f"Retrieved Knowledge Snippet: {state['retrieved_knowledge'][:200]}...")

        except Exception as e:
            logger.error(f"Error searching knowledge base: {type(e).__name__}: {repr(e)}", exc_info=True)
            state["knowledge_sources"] = []
            state["retrieved_knowledge"] = ""
        state["current_step"] = "hypothesis_generation"
        return state

    @traceable(name="Step 3: Hypothesis Generation")
    def _hypothesis_generation_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        logger.info("Executing Step 3: Hypothesis Generation")
        
        differential = self.hypothesis_generation_agent.invoke({
            "assessment": state["symptom_analysis"],
            "retrieved_knowledge": state["retrieved_knowledge"]
        })
        state["differential_diagnosis"] = differential.model_dump()
        
        logger.info(f"Generated Differential Diagnosis: {state['differential_diagnosis']}")
        
        state["current_step"] = "clarifying_questions"
        return state

    @traceable(name="Step 4: Clarifying Questions")
    def _clarifying_questions_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        logger.info("Executing Step 4: Clarifying Questions")
        
        # Initialize user_answers if not present
        if "previously_asked_questions" not in state:
            state["previously_asked_questions"] = []
            
        if "question_topics_covered" not in state:
            state["question_topics_covered"] = set()
        
        previously_asked=[]
        for q_data in state["previously_asked_questions"]:
            if isinstance(q_data, dict) and "text" in q_data:
                previously_asked.append(q_data["text"])
                logger.info(f"Previously Asked Questions: {previously_asked}")
                
            
        # if not state.get("user_answers"):
        #     state["user_answers"] = {}
        # if not state.get("question_count"):
        #     state["question_count"] = 0
        
        # Generate clarifying questions with metadata for UI rendering
        questions = self.clarifying_question_agent.invoke({
            "differential_diagnosis": state["differential_diagnosis"],
            "assessment": state["symptom_analysis"],
            "user_answers": state.get("user_answers", {}),
            "question_count": state.get("question_count", 0),
            "previously_asked": previously_asked

        })
        
        questions_data=questions.model_dump()
        new_questions = questions_data.get("questions", [])
        
        state["previously_asked_questions"].extend(new_questions)
        for q in new_questions:
            topic=q.get("id","").split("_")[0]
            state["question_topics_covered"].add(topic)

        state["questions_asked"] = questions_data
        state["question_count"]=state.get("question_count",0) +1
        
        
        # state["questions_asked"] = questions.model_dump()
        # logger.info(f"Generated Clarifying Questions: {state['questions_asked']}")
        
        # Increment question count
        #state["question_count"] += 1
        
        return state

    def _should_continue_questioning(self, state: MedicalDiagnosisState) -> str:
        """
        Conditional routing function - decides whether to continue asking questions
        or move to hypothesis refinement
        """
        logger.info("Evaluating whether to continue questioning...")
        
        question_count = state.get("question_count", 0)
        user_answers = state.get("user_answers", {})
        topics_covered = state.get("question_topics_covered", set())
        
        # Continue if we have less than 5 questions (minimum required)
        if question_count < 5:
            logger.info(f"📝 Continue - minimum questions not reached ({question_count}/5)")
            return "continue"
        
        # Count meaningful answers (non-empty, non-skip)
        meaningful_answers = len([a for a in user_answers.values() if a and str(a).strip() and str(a) != "skip"])
        
        # Stop if we have enough comprehensive information (15+ questions or 10+ meaningful answers)
        if question_count >= 15 or meaningful_answers >= 10:
            logger.info(f"✅ Sufficient information collected ({question_count} questions, {meaningful_answers} answers)")
            return "finish"
        
        # Check essential topics coverage
        essential_topics = {"pain", "duration", "location", "severity", "timing", "associated", "triggers"}
        covered_essential = len(essential_topics.intersection(topics_covered))
        
        # Stop if we have good coverage of essential topics and reasonable answers
        if covered_essential >= 5 and meaningful_answers >= 7:
            logger.info(f"✅ Essential topics covered ({covered_essential}/7) with {meaningful_answers} answers")
            return "finish"
        
        # Otherwise continue asking questions
        logger.info(f"📝 Continue - need more coverage (topics: {covered_essential}, answers: {meaningful_answers})")
        return "continue"
        # Check if we have enough information to proceed
        # questions_data = state.get("questions_asked", {})
        # questions_list = questions_data.get("questions", [])
        
        # Find unanswered questions
        # unanswered_questions = []
        # for q in questions_list:
        #     question_id = q.get("id", q.get("question", ""))
        #     if question_id not in state.get("user_answers", {}):
        #         unanswered_questions.append(q)
        
        # # Check if we need more questions (5-25 range based on complexity)
        # total_answers = len(state.get("user_answers", {}))
        # min_questions = 5
        # max_questions = 25
        
        # Decision logic
        # if total_answers < min_questions:
        #     logger.info(f"Need more questions: {total_answers}/{min_questions} minimum")
        #     return "continue"
        
        # if total_answers >= max_questions:
        #     logger.info(f"Maximum questions reached: {total_answers}/{max_questions}")
        #     return "finish"
        
        # # Check if we have unanswered questions in current batch
        # if unanswered_questions:
        #     logger.info(f"Still have {len(unanswered_questions)} unanswered questions")
        #     return "continue"
        
        # # Intelligent evaluation: check if key areas are covered
        # if self._has_sufficient_diagnostic_info(state):
        #     logger.info("Sufficient diagnostic information gathered")
        #     return "finish"
        
        # # Need more questions
        # logger.info("Need additional questions for complete diagnosis")
        # return "continue"
    
    def _deduplicate_questions(self, new_questions: List[dict], previous_questions: List[dict]) -> List[dict]:
        """Remove duplicate questions based on text similarity"""
        
        if not previous_questions:
            return new_questions
        
        # Extract previous question texts
        previous_texts = set()
        for pq in previous_questions:
            if isinstance(pq, dict) and "text" in pq:
                previous_texts.add(pq["text"].lower().strip())
        
        # Filter out duplicates and similar questions
        filtered_questions = []
        for q in new_questions:
            q_text = q.get("text", "").lower().strip()
            
            # Check for exact match
            if q_text in previous_texts:
                logger.info(f"Skipping duplicate question: {q_text}")
                continue
            
            # Check for high similarity
            is_similar = any(
                self._calculate_similarity(q_text, prev_text) > 0.8 
                for prev_text in previous_texts
            )
            
            if not is_similar:
                filtered_questions.append(q)
            else:
                logger.info(f"Skipping similar question: {q_text}")
        
        return filtered_questions

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
            
    def _has_sufficient_diagnostic_info(self, state: MedicalDiagnosisState) -> bool:
        """Check if we have sufficient information for diagnosis"""
        user_answers = state.get("user_answers", {})
        
        # Check if we've covered critical diagnostic areas
        critical_areas = ["severity", "duration", "location", "triggers", "associated_symptoms"]
        covered_areas = []
        
        for answer_key in user_answers.keys():
            for area in critical_areas:
                if area.lower() in answer_key.lower():
                    covered_areas.append(area)
        
        # If we've covered at least 3 critical areas and have minimum answers
        unique_areas = set(covered_areas)
        return len(unique_areas) >= 3 and len(user_answers) >= 5

    def add_user_answer(self, question_id: str, answer, state: MedicalDiagnosisState):
        """Add a user answer to the state"""
        if not state.get("user_answers"):
            state["user_answers"] = {}
        
        state["user_answers"][question_id] = answer
        logger.info(f"Added user answer for {question_id}: {answer}")
        
        return state

    def get_next_question(self, state: MedicalDiagnosisState):
        """Get the next unanswered question from the current question set"""
        questions_data = state.get("questions_asked", {})
        questions_list = questions_data.get("questions", [])
        
        for q in questions_list:
            question_id = q.get("id", q.get("question", ""))
            if question_id not in state.get("user_answers", {}):
                return q
        
        return None

    def start_interactive_diagnosis(self, symptoms: str, patient_info: dict = None):
        """Start an interactive diagnosis session that returns questions for UI"""
        logger.info(f"🩺 Starting interactive medical diagnosis for: {symptoms[:50]}...")
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
            current_step="initial_assessment",
            question_count=0,
            pending_question={}
        )
        
        # Run workflow until we hit a point where user input is needed
        try:
            # Run initial steps
            state = self._initial_assessment_step(initial_state)
            state = self._information_gathering_step(state)
            state = self._hypothesis_generation_step(state)
            state = self._clarifying_questions_step(state)
            
            # Return the first question for the UI
            next_question = self.get_next_question(state)
            if next_question:
                state["pending_question"] = next_question
                return {
                    "status": "question_pending",
                    "question": next_question,
                    "state": state
                }
            else:
                # No questions needed, continue to diagnosis
                return self.complete_diagnosis(state)
                
        except Exception as e:
            logger.error(f"❌ Error in interactive diagnosis: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}

    def answer_question(self, question_id: str, answer, state: MedicalDiagnosisState):
        """Process a user answer and return next question or diagnosis"""
        try:
            # Add the answer to state
            state = self.add_user_answer(question_id, answer, state)
            
            # Check if more questions are needed
            should_continue = self._should_continue_questioning(state)

            if should_continue == "continue":
                # Generate more questions
                state = self._clarifying_questions_step(state)
                next_question = self.get_next_question(state)
                
                if next_question:
                    state["pending_question"] = next_question
                    return {
                        "status": "question_pending",
                        "question": next_question,
                        "state": state
                    }
            
            # No more questions needed, complete diagnosis
            return self.complete_diagnosis(state)
            
        except Exception as e:
            logger.error(f"❌ Error processing answer: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}

    def complete_diagnosis(self, state: MedicalDiagnosisState):
        """Complete the diagnosis workflow and return final results"""
        try:
            logger.info("🏁 Starting complete_diagnosis - running remaining workflow steps...")
            
            # Run remaining steps
            state = self._hypothesis_refinement_step(state)
            state = self._final_diagnosis_step(state)
            state = self._treatment_plan_step(state)
            
            logger.info("✅ All workflow steps completed successfully")
            logger.info(f"📊 Final diagnosis: {state.get('final_diagnosis', {}).get('primary_diagnosis', 'Unknown')}")
            logger.info(f"🎯 Confidence score: {state.get('confidence_score', 0.0)}")
            
            result = {
                "status": "diagnosis_complete",
                "final_diagnosis": state.get("final_diagnosis", {}),
                "medications": state.get("medications", {}),
                "confidence_score": state.get("confidence_score", 0.0),
                "knowledge_sources": state.get("knowledge_sources", []),
                "state": state
            }
            
            logger.info("🎉 Returning diagnosis_complete status to UI")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error completing diagnosis: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}

    @traceable(name="Step 5: Hypothesis Refinement")
    def _hypothesis_refinement_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        logger.info("Executing Step 5: Hypothesis Refinement")
        refined = self.hypothesis_refinement_agent.invoke({
            "differential_diagnosis": state["differential_diagnosis"],
            "user_answers": state["user_answers"]
        })
        state["differential_diagnosis"] = refined.model_dump()
        logger.info(f"Refined Diagnosis: {state['differential_diagnosis']}")
        state["current_step"] = "final_diagnosis"
        return state

    @traceable(name="Step 6: Final Diagnosis")
    def _final_diagnosis_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        logger.info("Executing Step 6: Final Diagnosis")
        final = self.final_diagnosis_agent.invoke({
            "refined_diagnosis": state["differential_diagnosis"]
        })
        state["final_diagnosis"] = final.model_dump()
        state["confidence_score"] = final.confidence_score
        logger.info(f"Final Diagnosis: {state['final_diagnosis']} with confidence {state['confidence_score']}")
        state["current_step"] = "treatment_plan"
        return state

    @traceable(name="Step 7: Treatment Plan")
    def _treatment_plan_step(self, state: MedicalDiagnosisState) -> MedicalDiagnosisState:
        logger.info("Executing Step 7: Treatment Plan")
        plan = self.treatment_plan_agent.invoke({
            "final_diagnosis": state["final_diagnosis"],
            "retrieved_knowledge": state["retrieved_knowledge"]
        })
        state["medications"] = plan.model_dump()
        logger.info(f"Generated Treatment Plan: {state['medications']}")
        state["current_step"] = "complete"
        return state

    @traceable(name="Medical Diagnosis Workflow")
    def run_diagnosis(self, symptoms: str, patient_info: dict = None) -> dict:
        logger.info(f"🩺 Starting medical diagnosis for: {symptoms[:50]}...")
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
            current_step="initial_assessment",
            question_count=0,
            pending_question={}
        )
        try:
            result = self.app.invoke(initial_state)
            logger.info("✅ Diagnosis workflow completed")
            if result is None:
                logger.error("Workflow returned None")
                return {"error": "Workflow returned None", "final_diagnosis": {"primary_diagnosis": "System Error", "confidence_score": 0.0}, "confidence_score": 0.0}
            return result
        except Exception as e:
            logger.error(f"❌ Error in diagnosis workflow: {e}", exc_info=True)
            return {"error": str(e), "final_diagnosis": {"primary_diagnosis": "System Error", "confidence_score": 0.0}, "confidence_score": 0.0}


