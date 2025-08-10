"""
Agent: Final Diagnosis
Description: Provides a final, reasoned diagnosis with a confidence score and 
             recommendations for next steps.
"""
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain.output_parsers import PydanticOutputParser
except Exception:
    from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

from llm.llm_config import get_llm
from agents.hypothesis_refinement_agent import RefinedDifferentialDiagnosis  # noqa: F401 (type reference in docstrings)

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
            
            Your tone should be empathetic, clear, and highly responsible.
            
            Output format (must strictly follow):
            {format_instructions}
            """,
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
    parser = PydanticOutputParser(pydantic_object=FinalDiagnosis)
    prompt = FINAL_DIAGNOSIS_PROMPT.partial(format_instructions=parser.get_format_instructions())
    agent = prompt | llm | parser
    return agent

# --- Example Usage (for testing) ---

if __name__ == '__main__':
    pass
