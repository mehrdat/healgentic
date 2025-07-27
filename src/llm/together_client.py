"""
Together AI client for medical diagnosis
"""

import os
from typing import List
from together import Together
from dotenv import load_dotenv


class TogetherMedicalLLM:
    """Medical LLM using Together AI"""
    
    def __init__(self):
        load_dotenv()
        self.client = Together()
        self.model = os.getenv("TOGETHER_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")
    
    def analyze_symptoms(self, symptoms: str, medical_context: str = "") -> str:
        """Analyze symptoms using Together AI"""
        prompt = f"""You are a medical AI assistant. Analyze the following symptoms and provide a detailed medical assessment.

SYMPTOMS: {symptoms}

MEDICAL CONTEXT: {medical_context}

Please provide:
1. Symptom analysis
2. Possible conditions
3. Recommended follow-up questions
4. Severity assessment

Format your response as structured medical analysis."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def generate_differential_diagnosis(self, symptoms: str, context: str = "") -> str:
        """Generate differential diagnosis"""
        prompt = f"""As a medical expert, create a differential diagnosis for these symptoms:

SYMPTOMS: {symptoms}
CONTEXT: {context}

Provide a ranked list of possible diagnoses with:
1. Most likely diagnosis
2. Alternative diagnoses
3. Reasoning for each
4. Required tests or examinations

Be thorough but concise."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Diagnosis error: {str(e)}"

    def generate_questions(self, current_info: str) -> str:
        """Generate follow-up medical questions"""
        prompt = f"""Based on the current medical information, generate 3-5 specific follow-up questions to help narrow down the diagnosis.

CURRENT INFORMATION: {current_info}

Generate questions that are:
1. Specific and relevant
2. Help differentiate between conditions
3. Easy for patients to understand
4. Clinically significant

Format as numbered list."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Question generation error: {str(e)}"
    
    def recommend_medications(self, diagnosis: str, context: str = "") -> str:
        """Generate medication recommendations"""
        prompt = f"""Based on the diagnosis, provide medication recommendations:

DIAGNOSIS: {diagnosis}
CONTEXT: {context}

Provide:
1. First-line medications
2. Dosage recommendations
3. Important side effects
4. Contraindications
5. Monitoring requirements

Include appropriate medical disclaimers."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Medication recommendation error: {str(e)}"
