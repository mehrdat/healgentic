"""
Medical Diagnosis Agents - New Architecture
"""

from .initial_assessment_agent import get_initial_assessment_agent, InitialQuery, StructuredAssessment
from .information_gathering_agent import get_information_gathering_agent
from .hypothesis_generation_agent import get_hypothesis_generation_agent
from .clarifying_question_agent import get_clarifying_question_agent
from .hypothesis_refinement_agent import get_hypothesis_refinement_agent
from .final_diagnosis_agent import get_final_diagnosis_agent
from .treatment_plan_agent import get_treatment_plan_agent

__all__ = [
    "get_initial_assessment_agent",
    "get_information_gathering_agent",
    "get_hypothesis_generation_agent", 
    "get_clarifying_question_agent",
    "get_hypothesis_refinement_agent",
    "get_final_diagnosis_agent",
    "get_treatment_plan_agent",
    "InitialQuery",
    "StructuredAssessment"
]
