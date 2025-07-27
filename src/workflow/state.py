"""
State definition for the medical diagnosis system
"""

from typing import Dict, List, TypedDict, Any


class MedicalDiagnosisState(TypedDict):
    """State shared between all agents in the medical diagnosis workflow"""
    
    # Input
    user_symptoms: str
    patient_info: Dict[str, Any]
    
    # Analysis results
    symptom_analysis: Dict[str, Any]  # From initial_assessment_agent
    differential_diagnosis: Dict[str, Any]  # From hypothesis_generation_agent
    questions_asked: Dict[str, Any]  # From clarifying_question_agent
    user_answers: Dict[str, str]  # User's answers to questions
    
    # Final results
    final_diagnosis: Dict[str, Any]  # From final_diagnosis_agent
    medications: Dict[str, Any]  # From treatment_plan_agent
    confidence_score: float
    
    # Knowledge base interaction
    knowledge_sources: List[str]
    retrieved_knowledge: str
    
    # Workflow metadata
    current_step: str
