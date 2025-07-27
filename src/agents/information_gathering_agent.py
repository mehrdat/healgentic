"""
Agent: Information Gathering
Description: Takes a structured assessment and generates targeted search queries 
             to retrieve relevant information from the medical knowledge base.
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from llm.llm_config import get_llm
from agents.initial_assessment_agent import StructuredAssessment

# --- Pydantic Models ---

class SearchQuery(BaseModel):
    """A search query to be used with a vector database."""
    query: str = Field(description="A concise, targeted search query for a medical vector database.")

class SearchQueries(BaseModel):
    """A list of search queries based on the patient's symptoms."""
    queries: List[SearchQuery] = Field(description="A list of 3-5 targeted search queries to find relevant medical information.")

# --- Prompt Template ---

INFORMATION_GATHERING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert medical researcher AI. Your task is to formulate precise search queries to find information relevant to a patient's symptoms.
            
            Based on the structured assessment provided, generate a list of 3-5 distinct search queries. These queries should be optimized for a medical vector database to find information about potential causes, related symptoms, and diagnostic criteria.
            
            Focus on creating queries that are specific and cover different aspects of the patient's condition. For example, instead of just "headache," a better query might be "causes of headache behind the eyes with nausea."
            
            Combine symptoms where it makes sense to do so.
            
            Generate ONLY the list of search queries.""",
        ),
        (
            "human",
            "Here is the structured assessment of the patient's condition:\n\n---\n"
            "Main Symptoms: {main_symptoms}\n"
            "Secondary Symptoms: {secondary_symptoms}\n"
            "Duration: {duration_of_symptoms}\n"
            "Patient Age: {patient_age}\n"
            "Patient Sex: {patient_sex}\n"
            "Other Info: {other_relevant_info}\n"
            "---",
        ),
    ]
)

# --- Agent Definition ---

def get_information_gathering_agent():
    """
    Creates and returns the information gathering agent.
    
    This agent takes a structured assessment and generates a list of
    optimized search queries for the knowledge base.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(SearchQueries)
    agent = INFORMATION_GATHERING_PROMPT | structured_llm
    return agent

# --- Example Usage (for testing) ---

if __name__ == '__main__':
    information_agent = get_information_gathering_agent()
    
    # Example from the initial assessment agent
    test_assessment = StructuredAssessment(
        main_symptoms=['headache behind eyes'],
        secondary_symptoms=['nausea', 'dizziness when standing up'],
        duration_of_symptoms='1 week',
        patient_age=45,
        patient_sex='male',
        other_relevant_info='No fever reported.',
        initial_summary="Patient presents with a week-long headache with nausea and dizziness."
    )
    
    response = information_agent.invoke(test_assessment.dict())
    
    print("--- Structured Assessment Input ---")
    print(test_assessment.dict())
    print("\n--- Generated Search Queries ---")
    for q in response.queries:
        print(f"- {q.query}")

    # Example 2
    test_assessment_2 = StructuredAssessment(
        main_symptoms=['sore throat', 'cough'],
        secondary_symptoms=[],
        duration_of_symptoms=None,
        patient_age=None,
        patient_sex=None,
        other_relevant_info=None,
        initial_summary="Patient has a sore throat and cough."
    )
    
    response_2 = information_agent.invoke(test_assessment_2.dict())
    print("\n--- Generated Search Queries (Example 2) ---")
    for q in response_2.queries:
        print(f"- {q.query}")
