"""
Agent: Interactive Clarifying Question Generation
Description: Generates targeted questions with UI metadata for the user to help 
            differentiate between the hypotheses in the differential diagnosis.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
try:
    from langchain.output_parsers import PydanticOutputParser
except Exception:
    from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import json
import re

from llm.llm_config import get_llm
# from agents.hypothesis_generation_agent import DifferentialDiagnosis  # not required at runtime

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
        description="A list of 3-8 targeted questions with UI metadata to help refine the diagnosis. Generate at least 5 questions in the first round."
    )
    more_needed: bool = Field(
        default=True, 
        description="Whether more questions will be needed after these are answered"
    )

# --- Prompt Template ---

CLARIFYING_QUESTIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are a medical expert generating clarifying questions for diagnosis.
    
    CRITICAL INSTRUCTIONS:
    - Look at previously_asked_questions and user_answers to see what has ALREADY been covered
    - DO NOT repeat or rephrase questions that have been asked before
    - Generate only NEW questions that explore DIFFERENT diagnostic aspects
    - If sufficient information has been gathered (15+ total questions), set more_needed to false
    
    QUESTION GENERATION RULES:
    - Generate 5-8 new questions per round (minimum 5 for first round)
    - Focus on unexplored diagnostic areas
    - Avoid redundant questions about severity, location, timing if already asked
    - Consider the differential diagnosis and what specific information would help narrow it down
    - For each question, specify the appropriate UI widget type:
                - "slider": For severity, pain levels (0-10 scale)
                - "select": For single choice from predefined options
                - "multiselect": For multiple symptoms/choices
                - "radio": For yes/no or small option sets
                - "number": For numeric values (age, count, etc.)
                - "text": For open-ended descriptions
                - "date": For dates and timeframes
    Previously asked: {previously_asked}
    Current question count: {question_count}
    
    Output format (must strictly follow):
    {format_instructions}
    """),
    ("human", """
    Differential diagnosis: {differential_diagnosis}
    Symptom analysis: {assessment}
    Current user answers: {user_answers}
    
    Generate new, non-repetitive questions to clarify the diagnosis.
    """)
])

# Add this to your clarifying_question_agent.py file

INTERACTIVE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are a medical expert generating clarifying questions for diagnosis.
    
    CRITICAL INSTRUCTIONS:
    - Look at previously_asked_questions and user_answers to see what has ALREADY been covered
    - DO NOT repeat or rephrase questions that have been asked before
    - Generate only NEW questions that explore DIFFERENT diagnostic aspects
    - If sufficient information has been gathered, set more_needed to false
    
    QUESTION GENERATION RULES:
    - Maximum 1-2 new questions per round
    - Focus on unexplored diagnostic areas
    - Avoid redundant questions about severity, location, timing if already asked
    - Consider the differential diagnosis and what specific information would help narrow it down
    
    Previously asked questions (DO NOT REPEAT): {previously_asked}
    Current question count: {question_count}
    User answers so far: {user_answers}
    
    Generate NEW, non-repetitive questions with appropriate UI metadata:
    - type: "text" (text input), "slider" (0-10 scale), "select" (dropdown), "multiselect", "number", "radio"
    - For sliders: include min, max values
    - For select/radio: include options list
    - For each question: provide clear reasoning why this question helps narrow the diagnosis
    """),
    ("human", """
    Differential diagnosis to investigate: {differential_diagnosis}
    
    Initial symptom analysis: {assessment}
    
    Generate clarifying questions that will help distinguish between the possible diagnoses.
    Focus on questions that haven't been asked yet and provide the most diagnostic value.
    """)
])

# --- Agent Definition ---

def get_clarifying_question_agent():
    """
    Creates and returns the interactive clarifying question agent.
    
    This agent takes a differential diagnosis and generates targeted questions
    with UI metadata for interactive answering.
    """
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=InteractiveClarifyingQuestions)
    prompt = CLARIFYING_QUESTIONS_PROMPT.partial(format_instructions=parser.get_format_instructions())

    text_out = StrOutputParser()

    def _safe_parse(s: str) -> InteractiveClarifyingQuestions:
        try:
            if not s or not str(s).strip():
                raise ValueError("empty output")
            # Try direct Pydantic parsing first
            return parser.parse(s)
        except Exception:
            # Try to recover JSON block from text
            try:
                m = re.search(r"\{[\s\S]*\}$", s.strip())
                if m:
                    data = json.loads(m.group(0))
                    return InteractiveClarifyingQuestions.model_validate(data)
            except Exception:
                pass
            # Fallback to a minimal valid default to keep flow alive
            default = {
                "questions": [
                    {
                        "id": "q_severity",
                        "text": "On a scale of 0-10, how severe is the main symptom?",
                        "type": "slider",
                        "reasoning": "Severity helps prioritize differential diagnoses.",
                        "min": 0,
                        "max": 10,
                        "default": 5,
                        "required": True
                    },
                    {
                        "id": "q_timing",
                        "text": "When did the symptoms start and are they constant or intermittent?",
                        "type": "text",
                        "reasoning": "Timing and pattern guide likely causes.",
                        "required": True
                    },
                    {
                        "id": "q_triggers",
                        "text": "Do certain foods, positions, or activities trigger or worsen it?",
                        "type": "multiselect",
                        "options": ["After meals", "At night", "Lying down", "Exercise", "Spicy foods", "None"],
                        "reasoning": "Triggers help distinguish between conditions.",
                        "required": False
                    }
                ],
                "more_needed": True
            }
            return InteractiveClarifyingQuestions.model_validate(default)

    agent = prompt | llm | text_out | RunnableLambda(_safe_parse)

    return agent

# --- Example Usage (for testing) ---

if __name__ == '__main__':
    # Minimal smoke test
    agent = get_clarifying_question_agent()
    sample = {
        "differential_diagnosis": {"hypotheses": [{"condition": "Migraine", "probability": 0.7, "reasoning": "Example"}]},
        "assessment": {"main_symptoms": ["headache"]},
        "user_answers": {},
        "question_count": 1,
        "previously_asked": []
    }
    try:
        out = agent.invoke(sample)
        print("OK", out)
    except Exception as e:
        print("Test failed:", e)
