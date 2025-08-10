"""
Agent: Treatment Plan
Description: Suggests potential next steps or general treatment options based on 
             the final diagnosis. This agent focuses on general advice and does 
             not prescribe medication.
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional

from llm.llm_config import get_llm
from agents.final_diagnosis_agent import FinalDiagnosis

# --- Pydantic Models ---

class TreatmentSuggestion(BaseModel):
    """A single suggestion for managing a condition."""
    suggestion: str = Field(description="A piece of general advice or a potential management strategy.")
    category: str = Field(description="The category of the suggestion (e.g., 'Lifestyle', 'Home Care', 'When to See a Doctor').")

class MedicationOption(BaseModel):
    """Comparable medication option with concise fields for UI tables."""
    name: str = Field(description="Medication or supplement name (generic preferred)")
    family: str = Field(description="Pharmacologic class or supplement category (e.g., PPI, H2 blocker, Alginate)")
    ranking: int = Field(ge=1, le=5, description="1-5 ranking for likelihood/fit")
    best_dose: str = Field(description="Typical best dose or dosing range in simple terms")
    otc_or_rx: str = Field(description="OTC or Rx")
    other_uses: List[str] = Field(default_factory=list, description="Other common indications/uses")
    one_line: str = Field(description="One-sentence explainer of why this fits")
    details: Optional[str] = Field(default=None, description="Short reasoning; mechanism/when to prefer/avoid")

class TreatmentPlan(BaseModel):
    """A set of general, non-prescriptive suggestions for the diagnosed condition."""
    condition: str = Field(description="The medical condition for which the plan is being generated.")
    suggestions: List[TreatmentSuggestion] = Field(description="A list of general suggestions for managing the condition.")
    medications: List[MedicationOption] = Field(default_factory=list, description="Comparable medication/supplement options with rankings and concise data for a table.")
    important_note: str = Field(description="A note emphasizing that these are general suggestions and a doctor should be consulted for a personal treatment plan.")

# --- Prompt Template ---

TREATMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a medical information AI. Your role is to provide general, non-prescriptive advice for a given medical condition based on information from a knowledge base.
            
                IMPORTANT: Provide concise, general information. Do NOT prescribe or give directives.
            
            Instructions:
                1) Provide 5-8 general suggestions (Lifestyle/Home Care/Monitoring/When to See a Doctor).
                2) Provide a compact comparison list of 4-8 medications/supplements suitable for this condition with fields:
                    - name, family, ranking (1-5), best_dose (concise), otc_or_rx, other_uses (2-4), one_line, details (short).
                3) Use generic names where possible. Avoid redundant or trivial tips.
                4) Keep text concise; one-sentence one_line per medication.
                5) Do not include disclaimers beyond important_note field.
            
            Example categories:
            - Home Care (e.g., 'Rest in a quiet, dark room.')
            - Lifestyle (e.g., 'Stay hydrated by drinking plenty of water.')
            - Monitoring (e.g., 'Keep a journal of your symptoms.')
            - When to See a Doctor (e.g., 'Consult a doctor if the headache becomes the worst you have ever experienced.')
            
            Generate ONLY the treatment plan in the requested structured fields.""",
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
