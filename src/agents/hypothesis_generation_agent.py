"""
Agent: Hypothesis Generation
Description: Takes the initial symptoms and retrieved knowledge to create a 
             differential diagnosis (a list of possible conditions).
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from llm.llm_config import get_llm
from agents.initial_assessment_agent import StructuredAssessment

# --- Pydantic Models ---

class DiagnosisHypothesis(BaseModel):
    """A single hypothesis for a medical diagnosis."""
    condition: str = Field(description="The name of the possible medical condition.")
    probability: float = Field(description="A score from 0.0 to 1.0 indicating the likelihood of this condition, based on the initial evidence.")
    reasoning: str = Field(description="A brief explanation of why this condition is a possibility, citing specific symptoms and retrieved knowledge.")

class DifferentialDiagnosis(BaseModel):
    """A list of possible medical conditions that could explain the patient's symptoms."""
    hypotheses: List[DiagnosisHypothesis] = Field(description="A ranked list of the most likely medical conditions.")

# --- Prompt Template ---

HYPOTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert diagnostician AI. Your role is to analyze a patient's symptoms and the relevant medical information retrieved from a knowledge base to generate a differential diagnosis.
            
            A differential diagnosis is a list of possible conditions that could explain the patient's symptoms.
            
            Instructions:
            1.  Review the patient's structured assessment.
            2.  Carefully consider the provided context from the medical knowledge base.
            3.  Generate a list of the most likely diagnoses.
            4.  For each diagnosis, provide a probability score (0.0 to 1.0) and a brief, evidence-based reasoning. The probabilities should be your estimated likelihood based on the provided information and should ideally sum to a value close to 1.0 across all hypotheses.
            5.  The reasoning should connect the patient's symptoms to the diagnostic criteria mentioned in the retrieved knowledge.
            
            Your analysis must be based *only* on the information provided. Do not invent information.
            Present the most likely condition first.""",
        ),
        (
            "human",
            "Here is the patient's information and the relevant medical knowledge:\n\n"
            "--- Patient Assessment ---\n"
            "{assessment}\n\n"
            "--- Retrieved Medical Knowledge ---\n"
            "{retrieved_knowledge}\n"
            "---",
        ),
    ]
)

# --- Agent Definition ---

def get_hypothesis_generation_agent():
    """
    Creates and returns the hypothesis generation agent.
    
    This agent takes the patient's assessment and retrieved knowledge
    and generates a differential diagnosis.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(DifferentialDiagnosis)
    agent = HYPOTHESIS_PROMPT | structured_llm
    return agent

# --- Example Usage (for testing) ---

if __name__ == '__main__':
    hypothesis_agent = get_hypothesis_generation_agent()
    
    test_assessment = StructuredAssessment(
        main_symptoms=['headache behind eyes'],
        secondary_symptoms=['nausea', 'dizziness when standing up'],
        duration_of_symptoms='1 week',
        patient_age=45,
        patient_sex='male',
        other_relevant_info='No fever reported.',
        initial_summary="Patient presents with a week-long headache with nausea and dizziness."
    )
    
    retrieved_knowledge = """
    - Migraine: Often characterized by severe, throbbing headache, frequently unilateral but can be bilateral. Often accompanied by nausea, vomiting, and sensitivity to light and sound. Can be triggered by stress or specific foods.
    - Tension Headache: Described as a constant ache or pressure around the head, especially at the temples or back of the head and neck. Not usually associated with nausea.
    - Cluster Headache: Severe, recurring headaches that are one-sided, often behind one eye. Attacks occur in clusters or cycles.
    - Postural Orthostatic Tachycardia Syndrome (POTS): A condition affecting blood flow, causing dizziness or fainting upon standing. Headaches are a common symptom. Nausea can also occur.
    """
    
    response = hypothesis_agent.invoke({
        "assessment": test_assessment.dict(),
        "retrieved_knowledge": retrieved_knowledge
    })
    
    print("--- Differential Diagnosis ---")
    for h in response.hypotheses:
        print(f"- Condition: {h.condition} (Probability: {h.probability:.2f})")
        print(f"  Reasoning: {h.reasoning}")
