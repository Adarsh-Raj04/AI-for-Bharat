"""
Regulatory Chain - Specialized chain for FDA/regulatory compliance queries
Phase 3 Feature #19
"""

from typing import Dict, Any, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)


class RegulatoryChain:
    """
    Specialized chain for regulatory and compliance information.
    Handles FDA approvals, guidelines, regulatory requirements, and compliance queries.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize regulatory prompts"""
        
        # FDA approval information
        self.fda_approval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a regulatory affairs expert specializing in FDA approvals and drug regulations.

TASK: Provide comprehensive FDA approval information.

REQUIRED STRUCTURE:

## Approval Summary
**Drug/Device**: [Name]
**Approval Date**: [Date] [citation]
**Regulatory Pathway**: [Pathway] [citation]
**Indication**: [Approved indication] [citation]

## Approval Details

### Regulatory Pathway
- **Type**: [Standard/Accelerated/Priority Review/Breakthrough] [citation]
- **Designation**: [Orphan Drug/Fast Track/etc.] [citation]
- **Review Timeline**: [Duration] [citation]

### Clinical Evidence
- **Pivotal Trials**: [Trial names/NCT numbers] [citation]
- **Primary Endpoint**: [Endpoint and results] [citation]
- **Patient Population**: [Description] [citation]
- **Efficacy Data**: [Key results] [citation]

### Safety Information
- **Boxed Warnings**: [If any] [citation]
- **Contraindications**: [List] [citation]
- **Warnings and Precautions**: [Key safety information] [citation]
- **Adverse Reactions**: [Common AEs] [citation]

### Post-Marketing Requirements
- **Confirmatory Trials**: [If accelerated approval] [citation]
- **REMS Program**: [If applicable] [citation]
- **Additional Studies**: [Required studies] [citation]

## Labeling Information
[Key points from FDA-approved labeling]

## Regulatory Notes
[Important regulatory considerations or updates]

INSTRUCTIONS:
1. Use ONLY information from the provided context
2. Include specific dates and regulatory body names
3. Cite all information with [number] notation
4. Distinguish between requirements and recommendations
5. Note any regional differences (FDA vs EMA vs other)
6. Include relevant guidance document references
7. Use proper markdown formatting for lists and sections
8. If using tables, ensure proper markdown table syntax with pipes (|) and dashes (-)

CONTEXT:
{context}"""),
            ("human", "{question}")
        ])
        
        # Regulatory guidance/requirements
        self.guidance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a regulatory affairs expert on FDA guidance and requirements.

TASK: Provide regulatory guidance and requirement information.

REQUIRED STRUCTURE:

## Guidance Overview
**Title**: [Guidance document title] [citation]
**Issuing Body**: [FDA/EMA/ICH/etc.] [citation]
**Date**: [Issue/revision date] [citation]
**Status**: [Draft/Final] [citation]

## Key Requirements

### Mandatory Requirements
1. [Requirement 1] [citation]
   - Details: [Explanation]
   - Applicability: [When this applies]

2. [Requirement 2] [citation]
   - Details: [Explanation]
   - Applicability: [When this applies]

### Recommendations
1. [Recommendation 1] [citation]
   - Rationale: [Why recommended]
   - Implementation: [How to implement]

## Compliance Considerations
- **Timeline**: [Implementation timeline] [citation]
- **Scope**: [What products/studies are covered] [citation]
- **Exceptions**: [Any exceptions or special cases] [citation]

## Regional Differences
[Note differences between FDA, EMA, and other regulatory bodies]

## References
[List relevant guidance documents and regulations]

INSTRUCTIONS:
1. Clearly distinguish between requirements and recommendations
2. Include specific regulation/guidance citations (21 CFR, ICH, etc.)
3. Note effective dates and transition periods
4. Cite sources with [number] notation
5. Be precise about regulatory language

CONTEXT:
{context}"""),
            ("human", "{question}")
        ])
        
        # Clinical trial regulatory requirements
        self.trial_regulatory_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a regulatory expert on clinical trial requirements.

TASK: Provide clinical trial regulatory requirement information.

REQUIRED STRUCTURE:

## Regulatory Framework
**Trial Phase**: [Phase] [citation]
**Regulatory Pathway**: [IND/IDE/etc.] [citation]
**Applicable Regulations**: [21 CFR parts, ICH guidelines] [citation]

## Pre-Trial Requirements

### IND/IDE Requirements
- **Application Type**: [Initial/Amendment] [citation]
- **Required Documentation**: [List] [citation]
- **Review Timeline**: [Duration] [citation]

### IRB/Ethics Committee
- **Approval Requirements**: [Details] [citation]
- **Continuing Review**: [Frequency] [citation]
- **Reporting Requirements**: [What must be reported] [citation]

## During-Trial Requirements

### Monitoring and Reporting
- **Safety Reporting**: [Timeline and requirements] [citation]
- **Protocol Amendments**: [When required] [citation]
- **Data Safety Monitoring**: [Requirements] [citation]

### GCP Compliance
- **Key Requirements**: [List] [citation]
- **Documentation**: [Essential documents] [citation]
- **Inspections**: [What to expect] [citation]

## Post-Trial Requirements
- **Final Report**: [Timeline and content] [citation]
- **Data Retention**: [Duration] [citation]
- **Publication Requirements**: [If any] [citation]

## Compliance Tips
[Practical guidance for maintaining compliance]

INSTRUCTIONS:
1. Be specific about regulatory citations
2. Include timelines and deadlines
3. Note differences by trial phase
4. Cite sources with [number] notation
5. Provide actionable guidance

CONTEXT:
{context}"""),
            ("human", "{question}")
        ])
        
        # General regulatory query
        self.general_regulatory_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a regulatory affairs expert.

TASK: Provide regulatory and compliance information.

REQUIRED STRUCTURE:

## Regulatory Overview
[Brief overview of the regulatory topic]

## Key Regulatory Points

### Requirements
- [Requirement 1] [citation]
- [Requirement 2] [citation]
- [Requirement 3] [citation]

### Guidelines and Recommendations
- [Guideline 1] [citation]
- [Guideline 2] [citation]

### Regulatory Bodies
- **FDA**: [Relevant FDA information] [citation]
- **EMA**: [Relevant EMA information] [citation]
- **Other**: [Other regulatory bodies] [citation]

## Compliance Considerations
[Important compliance points]

## Timeline and Deadlines
[If applicable]

## Additional Resources
[Relevant guidance documents, regulations, or resources]

INSTRUCTIONS:
1. Use ONLY information from the provided context
2. Include specific regulatory citations
3. Note dates and regulatory body names
4. Distinguish requirements from recommendations
5. Cite sources with [number] notation
6. Be precise with regulatory language

CONTEXT:
{context}"""),
            ("human", "{question}")
        ])
        
        logger.info("Initialized regulatory prompts")
    
    def detect_regulatory_type(self, query: str) -> str:
        """
        Detect what type of regulatory query this is
        
        Returns: 'approval', 'guidance', 'trial', or 'general'
        """
        query_lower = query.lower()
        
        # FDA approval indicators
        approval_keywords = [
            'approval', 'approved', 'fda approved', 'indication',
            'label', 'labeling', 'package insert', 'accelerated approval',
            'breakthrough', 'priority review', 'orphan drug'
        ]
        
        # Guidance/requirement indicators
        guidance_keywords = [
            'guidance', 'guideline', 'requirement', 'regulation',
            '21 cfr', 'ich', 'gcp', 'glp', 'compliance',
            'regulatory requirement'
        ]
        
        # Clinical trial regulatory indicators
        trial_keywords = [
            'ind', 'ide', 'clinical trial', 'irb', 'ethics committee',
            'protocol', 'informed consent', 'monitoring',
            'trial requirement', 'study requirement'
        ]
        
        approval_score = sum(1 for kw in approval_keywords if kw in query_lower)
        guidance_score = sum(1 for kw in guidance_keywords if kw in query_lower)
        trial_score = sum(1 for kw in trial_keywords if kw in query_lower)
        
        scores = {
            'approval': approval_score,
            'guidance': guidance_score,
            'trial': trial_score
        }
        
        max_type = max(scores, key=scores.get)
        if scores[max_type] > 0:
            return max_type
        return 'general'
    
    def get_regulatory_info(
        self,
        query: str,
        context: str,
        regulatory_type: Optional[str] = None
    ) -> str:
        """
        Generate regulatory information response
        
        Args:
            query: Regulatory query
            context: Formatted context documents
            regulatory_type: Type of regulatory query
            
        Returns:
            Structured regulatory information
        """
        try:
            # Auto-detect regulatory type if not provided
            if regulatory_type is None:
                regulatory_type = self.detect_regulatory_type(query)
            
            # Select appropriate prompt
            prompt_map = {
                'approval': self.fda_approval_prompt,
                'guidance': self.guidance_prompt,
                'trial': self.trial_regulatory_prompt,
                'general': self.general_regulatory_prompt
            }
            
            prompt = prompt_map.get(regulatory_type, self.general_regulatory_prompt)
            
            # Build chain
            chain = (
                {
                    "context": lambda _: context,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate response
            result = chain.invoke(query)
            logger.info(f"Generated {regulatory_type} regulatory information")
            return result
            
        except Exception as e:
            logger.error(f"Regulatory information generation error: {e}")
            return f"Error generating regulatory information: {str(e)}"


def get_regulatory_chain(llm) -> RegulatoryChain:
    """Create a regulatory chain instance"""
    return RegulatoryChain(llm)
