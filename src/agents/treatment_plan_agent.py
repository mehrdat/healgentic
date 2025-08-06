"""
Agent: Treatment Plan
Description: Suggests potential next steps or general treatment options based on 
             the final diagnosis. This agent focuses on general advice and does 
             not prescribe medication.
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from llm.llm_config import get_llm
from agents.final_diagnosis_agent import FinalDiagnosis

# --- Pydantic Models ---

class TreatmentSuggestion(BaseModel):
    """A single suggestion for managing a condition."""
    suggestion: str = Field(description="A piece of general advice or a potential management strategy.")
    category: str = Field(description="The category of the suggestion (e.g., 'Lifestyle', 'Home Care', 'When to See a Doctor').")

class TreatmentPlan(BaseModel):
    """A set of general, non-prescriptive suggestions for the diagnosed condition."""
    condition: str = Field(description="The medical condition for which the plan is being generated.")
    suggestions: List[TreatmentSuggestion] = Field(description="A list of general suggestions for managing the condition.")
    important_note: str = Field(description="A note emphasizing that these are general suggestions and a doctor should be consulted for a personal treatment plan.")

# --- Prompt Template ---

TREATMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a medical information AI. Your role is to provide general, non-prescriptive advice for a given medical condition based on information from a knowledge base.
            
            IMPORTANT: You must NOT prescribe medication or give definitive medical directives. Your advice should be general and focused on lifestyle, home care, and when to seek professional help.
            
            Instructions:
            1.  Based on the final diagnosis and the retrieved knowledge, provide a list of general suggestions.
            2.  Categorize each suggestion (e.g., 'Lifestyle', 'Home Care', 'When to See a Doctor').
            3.  Formulate an important note that stresses these are not personal medical instructions and a doctor must be consulted.
            4.  Your suggestions should be safe, widely accepted, and general in nature.
            
            Example categories:
            - Home Care (e.g., 'Rest in a quiet, dark room.')
            - Lifestyle (e.g., 'Stay hydrated by drinking plenty of water.')
            - Monitoring (e.g., 'Keep a journal of your symptoms.')
            - When to See a Doctor (e.g., 'Consult a doctor if the headache becomes the worst you have ever experienced.')
            
            Generate ONLY the treatment plan.""",
        ),
        (
            "human",
            "Here is the final diagnosis and the relevant medical knowledge:\n\n"
            "--- Final Diagnosis ---\n"
            "{final_diagnosis}\n\n"
            "--- Retrieved Medical Knowledge ---\n"
            "{retrieved_knowledge}\n"
            "---",
        ),
    ]
)

# --- Agent Definition ---

def get_treatment_plan_agent():
    """
    Creates and returns the treatment plan agent.
    
    This agent takes a final diagnosis and retrieved knowledge and suggests
    general, non-prescriptive next steps.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(TreatmentPlan)
    agent = TREATMENT_PROMPT | structured_llm
    return agent

# --- Example Usage (for testing) ---

if __name__ == '__main__':
    treatment_agent = get_treatment_plan_agent()
    
    final_diagnosis = FinalDiagnosis(
        primary_diagnosis="Migraine",
        confidence_score=0.9,
        final_summary="The diagnosis is likely a migraine, given the combination of a severe headache behind the eyes, nausea, and sensitivity to light.",
        next_steps=["Consult a doctor for a formal diagnosis.", "Keep a headache diary."],
        disclaimer="This is an AI-generated assessment and not a substitute for professional medical advice."
    )
    
    retrieved_knowledge = """
    - Migraine Management: Management often involves rest in a dark, quiet environment. Over-the-counter pain relievers can be effective for some. Staying hydrated is important. Identifying and avoiding triggers (like certain foods, stress, or lack of sleep) is a key long-term strategy. A doctor may prescribe specific medications like triptans. It is important to see a doctor if the headache pattern changes or if it is accompanied by a high fever or stiff neck.
    """
    
    response = treatment_agent.invoke({
        "final_diagnosis": final_diagnosis.dict(),
        "retrieved_knowledge": retrieved_knowledge
    })
    
    print(f"--- General Suggestions for: {response.condition} ---")
    for s in response.suggestions:
        print(f"- [{s.category}] {s.suggestion}")
    print(f"\nImportant Note: {response.important_note}")
