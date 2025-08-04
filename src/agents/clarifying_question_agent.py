"""
Agent: Interactive Clarifying Question Generation
Description: Generates targeted questions with UI metadata for the user to help 
            differentiate between the hypotheses in the differential diagnosis.
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional, Any

from llm.llm_config import get_llm
from agents.hypothesis_generation_agent import DifferentialDiagnosis

# --- Pydantic Models ---

class InteractiveClarifyingQuestion(BaseModel):
    """A single, targeted question with UI metadata for interactive answering."""
    id: str = Field(description="Unique identifier for this question")
    text: str = Field(description="The question to ask the user")
    type: str = Field(description="UI widget type: 'text', 'slider', 'select', 'multiselect', 'number', 'radio', 'date'")
    reasoning: str = Field(description="Why this question is being asked")
    
    # Widget-specific options
    options: Optional[List[str]] = Field(default=None, description="Options for select/multiselect/radio widgets")
    min: Optional[int] = Field(default=None, description="Minimum value for slider/number widgets")
    max: Optional[int] = Field(default=None, description="Maximum value for slider/number widgets")
    default: Optional[Any] = Field(default=None, description="Default value for the widget")
    required: bool = Field(default=True, description="Whether this question is required")

class InteractiveClarifyingQuestions(BaseModel):
    """A list of interactive clarifying questions to ask the user."""
    questions: List[InteractiveClarifyingQuestion] = Field(
        description="A list of 5-15 targeted questions with UI metadata to help refine the diagnosis"
    )
    more_needed: bool = Field(
        default=True, 
        description="Whether more questions will be needed after these are answered"
    )

# --- Prompt Template ---

INTERACTIVE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert medical diagnostician AI. Your goal is to formulate interactive clarifying questions to help differentiate between several possible medical conditions.
            
            You have been provided with a differential diagnosis and previous answers. Your task is to generate 3-8 specific, easy-to-understand questions for the patient that will help determine which diagnosis is the most accurate.
            
            Instructions:
            1. Analyze the list of possible conditions and previous answers.
            2. Identify the key differences in symptoms or characteristics between the top hypotheses.
            3. Formulate questions that directly probe these differences.
            4. For each question, specify the appropriate UI widget type:
                - "slider": For severity, pain levels (0-10 scale)
                - "select": For single choice from predefined options
                - "multiselect": For multiple symptoms/choices
                - "radio": For yes/no or small option sets
                - "number": For numeric values (age, count, etc.)
                - "text": For open-ended descriptions
                - "date": For dates and timeframes
            5. Provide appropriate options, min/max values for each widget.
            6. Generate questions that build on previous answers and avoid repetition.
            7. Focus on critical diagnostic factors: severity, duration, location, triggers, associated symptoms.
            
            Example question formats:
            - Severity: "How severe is your headache?" (slider, 0-10)
            - Location: "Where exactly is the pain located?" (select with body parts)
            - Duration: "How long have you had these symptoms?" (select with time ranges)
            - Associated symptoms: "Which of these symptoms do you also experience?" (multiselect)
            - Triggers: "What makes your symptoms worse?" (multiselect)
            
            Generate questions that will help narrow down from the current differential diagnosis.""",
        ),
        (
            "human",
            "Here is the current differential diagnosis:\n\n---\n"
            "{differential_diagnosis}\n"
            "---"
            "\nHere is the initial patient assessment:\n\n---\n"
            "{assessment}\n"
            "---"
            "\nPrevious user answers (avoid asking about these again):\n\n---\n"
            "{user_answers}\n"
            "---"
            "\nQuestion round: {question_count}\n\n"
            "Generate 5-15 targeted questions with appropriate UI widgets to help refine the diagnosis.",
        ),
    ]
)

# --- Agent Definition ---

def get_clarifying_question_agent():
    """
    Creates and returns the interactive clarifying question agent.
    
    This agent takes a differential diagnosis and generates targeted questions
    with UI metadata for interactive answering.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(InteractiveClarifyingQuestions)
    agent = INTERACTIVE_QUESTION_PROMPT | structured_llm
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
        "differential_diagnosis": test_diagnosis.model_dump(),
        "assessment": test_assessment.model_dump(),
        "user_answers": {},
        "question_count": 1
    })
    
    print("--- Generated Interactive Clarifying Questions ---")
    for q in response.questions:
        print(f"- ID: {q.id}")
        print(f"  Question: {q.text}")
        print(f"  Type: {q.type}")
        print(f"  Reasoning: {q.reasoning}")
        if q.options:
            print(f"  Options: {q.options}")
        if q.min is not None or q.max is not None:
            print(f"  Range: {q.min}-{q.max}")
        print()
