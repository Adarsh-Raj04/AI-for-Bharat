"""
Safety Guardrails - Detects and blocks unsafe medical advice queries
"""
import re
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SafetyGuardrails:
    """
    Detects queries requesting medical advice and blocks them
    Prevents the system from providing diagnostic or treatment recommendations
    """
    
    def __init__(self):
        self.diagnostic_patterns = self._build_diagnostic_patterns()
        self.treatment_patterns = self._build_treatment_patterns()
        self.personal_medical_patterns = self._build_personal_medical_patterns()
    
    def _build_diagnostic_patterns(self) -> list:
        """Patterns for diagnostic queries"""
        return [
            r'\b(do i have|am i|could i have|might i have)\b.*\b(disease|condition|disorder|syndrome|cancer|diabetes|infection)\b',
            r'\b(diagnose|diagnosis)\b.*\b(me|my|myself)\b',
            r'\b(what\'s wrong with|what is wrong with)\b.*\b(me|my)\b',
            r'\b(why (do|does)) (i|my)\b.*\b(have|feel|experience)\b',
            r'\b(is (this|it))\b.*\b(serious|dangerous|life-threatening|cancer|tumor)\b',
            r'\b(should i (be|get))\b.*\b(worried|concerned|tested|checked)\b',
            r'\b(my (symptoms?|pain|condition))\b.*\b(mean|indicate|suggest)\b',
        ]
    
    def _build_treatment_patterns(self) -> list:
        """Patterns for treatment recommendation queries"""
        return [
            r'\b(should i take|can i take|should i use)\b.*\b(drug|medication|medicine|pill|tablet)\b',
            r'\b(what (should|can) i (take|use|do))\b.*\b(for|to (treat|cure|fix))\b',
            r'\b(how (do|can) i (treat|cure|fix))\b.*\b(my|myself)\b',
            r'\b(is it safe (for me )?to (take|use))\b',
            r'\b(what (is|are) the (best|right))\b.*\b(treatment|medication|drug|therapy)\b.*\b(for me)\b',
            r'\b(should i (stop|start|continue))\b.*\b(taking|using|medication|drug|treatment)\b',
            r'\b(can i (stop|skip|miss))\b.*\b(dose|medication|pill|treatment)\b',
            r'\b(what (dose|dosage))\b.*\b(should i take)\b',
        ]
    
    def _build_personal_medical_patterns(self) -> list:
        """Patterns for personal medical situation queries"""
        return [
            r'\b(i (have|am|feel|experience|suffer from))\b',
            r'\b(my (doctor|physician|specialist))\b.*\b(said|told|prescribed|recommended)\b',
            r'\b(i (was|have been))\b.*\b(diagnosed|prescribed|told)\b',
            r'\b(i\'m (taking|on|using))\b.*\b(medication|drug|treatment|therapy)\b',
            r'\b(my (child|baby|son|daughter|parent|mother|father))\b.*\b(has|have|is)\b',
        ]
    
    def check_query(self, query: str) -> Tuple[bool, Optional[str], Dict]:
        """
        Check if query requests unsafe medical advice
        
        Args:
            query: User's query
            
        Returns:
            Tuple of (is_safe, block_reason, metadata)
        """
        query_lower = query.lower()
        
        # Check diagnostic patterns
        for pattern in self.diagnostic_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning(f"Blocked diagnostic query: {query[:100]}")
                return False, "diagnostic_request", {
                    "pattern_matched": pattern,
                    "query_type": "diagnostic"
                }
        
        # Check treatment patterns
        for pattern in self.treatment_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning(f"Blocked treatment query: {query[:100]}")
                return False, "treatment_request", {
                    "pattern_matched": pattern,
                    "query_type": "treatment"
                }
        
        # Check personal medical patterns (less strict, just flag)
        personal_match = False
        for pattern in self.personal_medical_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                personal_match = True
                break
        
        # If personal medical context detected, be extra cautious
        if personal_match:
            # Check if combined with diagnostic/treatment language
            risky_words = [
                'should i', 'can i', 'do i', 'am i', 'is it safe',
                'what should', 'how do i', 'help me', 'tell me if'
            ]
            
            for word in risky_words:
                if word in query_lower:
                    logger.warning(f"Blocked personal medical query: {query[:100]}")
                    return False, "personal_medical_advice", {
                        "query_type": "personal_medical"
                    }
        
        # Query is safe
        return True, None, {}
    
    def get_block_message(self, block_reason: str) -> str:
        """
        Get appropriate block message for the reason
        
        Args:
            block_reason: Reason for blocking
            
        Returns:
            User-friendly message
        """
        messages = {
            "diagnostic_request": """I cannot provide medical diagnoses or determine if you have a specific condition. 

**This system is designed for research purposes only and cannot:**
- Diagnose medical conditions
- Interpret your symptoms
- Determine if you have a disease

**Please consult a healthcare professional if you:**
- Have symptoms or health concerns
- Need a medical diagnosis
- Want to understand your health condition

**What I can help with:**
- Summarize published research papers
- Compare clinical trial results
- Explain medical concepts and terminology
- Provide information about FDA regulations

Would you like to rephrase your question to focus on research information instead?""",
            
            "treatment_request": """I cannot provide personal medical treatment recommendations or advice on medications.

**This system is designed for research purposes only and cannot:**
- Recommend specific treatments for you
- Advise on medication dosages
- Tell you whether to start, stop, or change medications
- Provide personalized medical advice

**Please consult a healthcare professional for:**
- Treatment recommendations
- Medication advice
- Dosage information
- Changes to your treatment plan

**What I can help with:**
- Summarize research on treatment approaches
- Compare clinical trial outcomes
- Explain how medications work (general information)
- Provide information about drug development

Would you like to rephrase your question to focus on research information instead?""",
            
            "personal_medical_advice": """I cannot provide advice about your personal medical situation.

**This system is designed for research purposes only and cannot:**
- Give advice about your specific health situation
- Interpret your symptoms or test results
- Recommend actions for your medical care

**Please consult your healthcare provider for:**
- Personal medical advice
- Questions about your diagnosis or treatment
- Concerns about your health

**What I can help with:**
- General research information
- Published study findings
- Medical terminology explanations
- Clinical trial data

Would you like to ask a general research question instead?"""
        }
        
        return messages.get(
            block_reason,
            "I cannot provide personal medical advice. Please consult a healthcare professional."
        )


# Singleton instance
_safety_guardrails = None


def get_safety_guardrails() -> SafetyGuardrails:
    """Get or create the safety guardrails singleton"""
    global _safety_guardrails
    if _safety_guardrails is None:
        _safety_guardrails = SafetyGuardrails()
    return _safety_guardrails
