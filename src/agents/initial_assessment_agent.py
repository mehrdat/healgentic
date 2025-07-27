"""
Agent: Initial Assessment
Description: Gathers and structures the initial user query (symptoms, age, etc.) 
            into a standardized format for the workflow.
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional

from llm.llm_config import get_llm

# --- Pydantic Models for Input and Output ---

class InitialQuery(BaseModel):
    """The initial, unstructured query from the user."""
    text: str = Field(description="The user's description of their symptoms and condition.")

class StructuredAssessment(BaseModel):
    """A structured representation of the patient's initial complaint."""
    main_symptoms: List[str] = Field(description="A list of the primary symptoms identified.")
    secondary_symptoms: List[str] = Field(description="A list of any secondary or less prominent symptoms mentioned.")
    duration_of_symptoms: Optional[str] = Field(description="The duration of the symptoms, if mentioned (e.g., '3 days', '2 weeks').")
    patient_age: Optional[int] = Field(description="The patient's age, if mentioned.")
    patient_sex: Optional[str] = Field(description="The patient's sex, if mentioned.")
    other_relevant_info: Optional[str] = Field(description="Any other potentially relevant information provided by the user.")
    initial_summary: str = Field(description="A concise one-sentence summary of the patient's chief complaint.")

# --- Prompt Template ---

ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert medical assistant AI. Your role is to carefully analyze a user's initial medical query and structure it into a standardized format. 
            
            Your primary goal is to extract key information accurately without making assumptions or providing medical advice.
            
            - Identify and list the main symptoms.
            - Identify and list any secondary symptoms.
            - Extract the duration of symptoms, patient age, and sex if provided.
            - Note any other information that seems relevant.
            - Create a very brief, one-sentence summary of the core issue.
            
            If a piece of information (like age or duration) is not mentioned, leave it as null.
            Focus ONLY on structuring the provided information.""",
        ),
        (
            "human",
            "Here is the user's query:\n\n---\n{text}\n---",
        ),
    ]
)

# --- Agent Definition ---

def get_initial_assessment_agent():
    """
    Creates and returns the initial assessment agent.
    
    This agent is a chain that takes an unstructured user query and returns a
    structured assessment of their condition.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(StructuredAssessment)
    agent = ASSESSMENT_PROMPT | structured_llm
    return agent

# --- Example Usage (for testing) ---

if __name__ == '__main__':
    assessment_agent = get_initial_assessment_agent()
    
    test_query = InitialQuery(
        text="Hi, I'm a 45-year-old male and for the past week I've had a really bad headache, mostly behind my eyes. I also feel a bit nauseous and dizzy, especially when I stand up too fast. I haven't had a fever."
    )
    
    response = assessment_agent.invoke(test_query.dict())
    
    print("--- Initial User Query ---")
    print(test_query.text)
    print("\n--- Structured Assessment ---")
    print(f"Main Symptoms: {response.main_symptoms}")
    print(f"Secondary Symptoms: {response.secondary_symptoms}")
    print(f"Duration: {response.duration_of_symptoms}")
    print(f"Age: {response.patient_age}")
    print(f"Sex: {response.patient_sex}")
    print(f"Other Info: {response.other_relevant_info}")
    print(f"Summary: {response.initial_summary}")

    # Example 2: Missing information
    test_query_2 = InitialQuery(
        text="I have got a extreme anxiety after using lamotrigine. after a year i still have the anxiety but i have another problem and tht is extreme reactin to medications and supplemets"
    )
    response_2 = assessment_agent.invoke(test_query_2.dict())
    print("\n--- Structured Assessment (Example 2) ---")
    print(f"Main Symptoms: {response_2.main_symptoms}")
    print(f"Secondary Symptoms: {response_2.secondary_symptoms}")
    print(f"Duration: {response_2.duration_of_symptoms}")
    print(f"Age: {response_2.patient_age}")
    print(f"Sex: {response_2.patient_sex}")
    print(f"Other Info: {response_2.other_relevant_info}")
    print(f"Summary: {response_2.initial_summary}")
