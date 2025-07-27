"""
Agent: Clarifying Question Generation
Description: Generates targeted questions for the user to help differentiate 
             between the hypotheses in the differential diagnosis.
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from llm.llm_config import get_llm
from agents.hypothesis_generation_agent import DifferentialDiagnosis

# --- Pydantic Models ---

class ClarifyingQuestion(BaseModel):
    """A single, targeted question to ask the user."""
    question: str = Field(description="The question to ask the user.")
    reasoning: str = Field(description="A brief explanation of why this question is being asked (i.e., what it helps to differentiate).")

class ClarifyingQuestions(BaseModel):
    """A list of clarifying questions to ask the user."""
    questions: List[ClarifyingQuestion] = Field(description="A list of 1-3 targeted questions to help refine the diagnosis.")

# --- Prompt Template ---

QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert medical diagnostician AI. Your goal is to formulate clarifying questions to help differentiate between several possible medical conditions.
            
            You have been provided with a differential diagnosis (a list of hypotheses). Your task is to generate 1-3 specific, easy-to-understand questions for the patient that will help determine which diagnosis is the most accurate.
            
            Instructions:
            1.  Analyze the list of possible conditions.
            2.  Identify the key differences in symptoms or characteristics between the top hypotheses.
            3.  Formulate questions that directly probe these differences. For example, if one condition involves a fever and another doesn't, ask about the patient's temperature.
            4.  For each question, provide a brief, internal reasoning for why the question is important.
            5.  Do not ask questions that have already been answered in the initial assessment.
            
            Generate ONLY the list of questions.""",
        ),
        (
            "human",
            "Here is the current differential diagnosis:\n\n---\n"
            "{differential_diagnosis}\n"
            "---"
            "\nHere is the initial patient assessment for context (do not ask about information already present here):\n\n---\n"
            "{assessment}\n"
            "---",
        ),
    ]
)

# --- Agent Definition ---

def get_clarifying_question_agent():
    """
    Creates and returns the clarifying question agent.
    
    This agent takes a differential diagnosis and generates targeted questions
    to ask the user.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(ClarifyingQuestions)
    agent = QUESTION_PROMPT | structured_llm
    return agent

# --- Example Usage (for testing) ---

if __name__ == '__main__':
    from medical_diagnosis_ai.src.agents.initial_assessment_agent import StructuredAssessment
    from medical_diagnosis_ai.src.agents.hypothesis_generation_agent import DiagnosisHypothesis

    question_agent = get_clarifying_question_agent()
    
    test_assessment = StructuredAssessment(
        main_symptoms=['headache behind eyes'],
        secondary_symptoms=['nausea', 'dizziness when standing up'],
        duration_of_symptoms='1 week',
        patient_age=45,
        patient_sex='male',
        other_relevant_info='No fever reported.',
        initial_summary="Patient presents with a week-long headache with nausea and dizziness."
    )

    test_diagnosis = DifferentialDiagnosis(
        hypotheses=[
            DiagnosisHypothesis(condition="Migraine", probability=0.7, reasoning="Headache with nausea fits the pattern of a migraine."),
            DiagnosisHypothesis(condition="Postural Orthostatic Tachycardia Syndrome (POTS)", probability=0.2, reasoning="Dizziness upon standing is a key indicator of POTS."),
            DiagnosisHypothesis(condition="Tension Headache", probability=0.1, reasoning="Headache is the primary symptom, but nausea is less common.")
        ]
    )
    
    response = question_agent.invoke({
        "differential_diagnosis": test_diagnosis.dict(),
        "assessment": test_assessment.dict()
    })
    
    print("--- Generated Clarifying Questions ---")
    for q in response.questions:
        print(f"- Question: {q.question}")
        print(f"  Reasoning: {q.reasoning}")
