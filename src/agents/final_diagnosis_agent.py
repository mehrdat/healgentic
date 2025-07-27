"""
Agent: Final Diagnosis
Description: Provides a final, reasoned diagnosis with a confidence score and 
             recommendations for next steps.
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from llm.llm_config import get_llm
from agents.hypothesis_refinement_agent import RefinedDifferentialDiagnosis

# --- Pydantic Models ---

class FinalDiagnosis(BaseModel):
    """The final output of the diagnostic process."""
    primary_diagnosis: str = Field(description="The name of the most likely medical condition.")
    confidence_score: float = Field(description="The final confidence score (from 0.0 to 1.0) in the primary diagnosis.")
    final_summary: str = Field(description="A comprehensive but easy-to-understand summary of the diagnostic process, explaining how the conclusion was reached.")
    next_steps: List[str] = Field(description="A list of recommended next steps for the user, such as 'Consult a doctor for a formal diagnosis' or 'Monitor symptoms for the next 24 hours'.")
    disclaimer: str = Field(description="A clear disclaimer that this is an AI-generated assessment and not a substitute for professional medical advice.")

# --- Prompt Template ---

FINAL_DIAGNOSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a senior medical diagnostician AI. Your final task is to synthesize all the information gathered and provide a clear, responsible, and well-reasoned final assessment.
            
            You have been provided with the refined differential diagnosis, which includes the most up-to-date probabilities for each condition.
            
            Instructions:
            1.  Identify the single most likely condition from the refined diagnosis list. This will be your primary diagnosis.
            2.  State your final confidence in this diagnosis as a score from 0.0 to 1.0. This should be based on the highest probability from the refined list.
            3.  Write a final summary that walks the user through the reasoning. Start with the initial symptoms, mention how the clarifying questions helped, and explain why the final diagnosis is the most probable one.
            4.  Provide a list of safe, responsible next steps. This should ALWAYS include a recommendation to consult a human doctor.
            5.  ALWAYS include a clear disclaimer that you are an AI and this is not a real medical diagnosis.
            
            Your tone should be empathetic, clear, and highly responsible.""",
        ),
        (
            "human",
            "Here is the refined differential diagnosis:\n\n---\n"
            "{refined_diagnosis}\n"
            "---",
        ),
    ]
)

# --- Agent Definition ---

def get_final_diagnosis_agent():
    """
    Creates and returns the final diagnosis agent.
    
    This agent takes the refined diagnosis and produces the final,
    user-facing output.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(FinalDiagnosis)
    agent = FINAL_DIAGNOSIS_PROMPT | structured_llm
    return agent

# --- Example Usage (for testing) ---

if __name__ == '__main__':
    from medical_diagnosis_ai.src.agents.hypothesis_generation_agent import DiagnosisHypothesis

    final_diagnosis_agent = get_final_diagnosis_agent()
    
    # This is the refined diagnosis from the previous agent's example
    refined_diagnosis = RefinedDifferentialDiagnosis(
        hypotheses=[
            DiagnosisHypothesis(condition="Migraine", probability=0.9, reasoning="Sensitivity to light is a classic migraine symptom, which the patient confirmed."),
            DiagnosisHypothesis(condition="Postural Orthostatic Tachycardia Syndrome (POTS)", probability=0.05, reasoning="Dizziness on standing is present, but the headache symptoms are more dominant."),
            DiagnosisHypothesis(condition="Tension Headache", probability=0.05, reasoning="This is now less likely given the confirmed sensitivity to light and nausea.")
        ],
        refinement_summary="The patient's report of sensitivity to light significantly increases the likelihood of a migraine and decreases the likelihood of a tension headache."
    )
    
    response = final_diagnosis_agent.invoke({
        "refined_diagnosis": refined_diagnosis.dict()
    })
    
    print("--- Final Diagnosis Output ---")
    print(f"Primary Diagnosis: {response.primary_diagnosis}")
    print(f"Confidence Score: {response.confidence_score:.2f}")
    print("\nFinal Summary:")
    print(response.final_summary)
    print("\nRecommended Next Steps:")
    for step in response.next_steps:
        print(f"- {step}")
    print(f"\nDisclaimer: {response.disclaimer}")
