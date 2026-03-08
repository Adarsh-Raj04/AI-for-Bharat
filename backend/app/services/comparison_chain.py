"""
Comparison Chain - Specialized chain for comparing drugs, treatments, or studies
Phase 3 Feature #16
"""

from typing import Dict, Any, List, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)


class ComparisonChain:
    """
    Specialized chain for systematic comparisons of drugs, treatments, or studies.
    Provides structured, table-based comparisons with evidence citations.
    """

    def __init__(self, llm):
        self.llm = llm
        self._initialize_prompts()

    def _initialize_prompts(self):
        """Initialize comparison prompts"""

        # Drug/Treatment comparison
        self.drug_comparison_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical research expert specializing in comparative analysis.

TASK: Compare the specified drugs or treatments systematically.

REQUIRED STRUCTURE:

## Comparison Overview
[Brief introduction stating what is being compared and why]

## Efficacy Comparison

| Metric | Item A | Item B | Difference |
|--------|--------|--------|------------|
| Response Rate | [X%] [citation] | [Y%] [citation] | [+/- Z%] |
| Median PFS | [X months] [citation] | [Y months] [citation] | [+/- Z months] |
| Median OS | [X months] [citation] | [Y months] [citation] | [+/- Z months] |
| ORR | [X%] [citation] | [Y%] [citation] | [+/- Z%] |

## Safety Comparison

| Adverse Event | Item A | Item B | Notes |
|---------------|--------|--------|-------|
| Grade 3+ AEs | [X%] [citation] | [Y%] [citation] | [Description] |
| Treatment Discontinuation | [X%] [citation] | [Y%] [citation] | [Reasons] |
| Common AEs | [List] [citation] | [List] [citation] | [Comparison] |

## Key Differences
- **Mechanism of Action**: [Compare MOA]
- **Dosing**: [Compare dosing schedules]
- **Patient Population**: [Compare approved indications]
- **Administration**: [Compare routes and frequency]

## Clinical Implications
[Evidence-based recommendations on when to use each option]

## Limitations
[Note any limitations in the comparison, such as lack of head-to-head trials]

INSTRUCTIONS:
1. Use ONLY information from the provided context
2. Present data in MARKDOWN TABLES with proper formatting (use | for columns, separate header with |---|---|)
3. Include specific numbers and statistics
4. Cite sources with [number] notation
5. Be objective and evidence-based
6. Note when direct comparisons are not available
7. Replace "Item A" and "Item B" with the actual names from the question
8. CRITICAL: Ensure all tables use proper markdown syntax with pipes (|) and dashes (-)
9. If any of the provided documents are not relevant to medical research, ignore them in the comparison and do not cite them and reply with "The provided context does not contain relevant information for this comparison.
10. If none of the provided documents are relevant to the comparison, state "The provided context does not contain relevant information for this comparison."

EXAMPLE TABLE FORMAT:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

CONTEXT:
{context}""",
                ),
                ("human", "{question}"),
            ]
        )

        # Study comparison
        self.study_comparison_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical research expert comparing clinical studies.

TASK: Compare the specified clinical studies systematically.

REQUIRED STRUCTURE:

## Studies Overview
[Brief description of each study being compared]

## Study Design Comparison

| Aspect | Study 1 | Study 2 | Notes |
|--------|---------|---------|-------|
| Phase | [Phase] [citation] | [Phase] [citation] | [Comparison] |
| Design | [Design] [citation] | [Design] [citation] | [Comparison] |
| Sample Size | [N] [citation] | [N] [citation] | [Comparison] |
| Population | [Description] [citation] | [Description] [citation] | [Comparison] |
| Primary Endpoint | [Endpoint] [citation] | [Endpoint] [citation] | [Comparison] |

## Results Comparison

| Outcome | Study 1 | Study 2 | Interpretation |
|---------|---------|---------|----------------|
| Primary Endpoint | [Result] [citation] | [Result] [citation] | [Analysis] |
| Secondary Endpoints | [Results] [citation] | [Results] [citation] | [Analysis] |
| Safety | [Profile] [citation] | [Profile] [citation] | [Analysis] |

## Methodological Differences
[Discuss key differences in methodology that may affect comparison]

## Conclusions
[Synthesize findings and note which study provides stronger evidence]

INSTRUCTIONS:
1. Use ONLY information from the provided context
2. Present data in MARKDOWN TABLES with proper formatting (use | for columns, separate header with |---|---|)
3. Include specific statistics and p-values
4. Cite sources with [number] notation
5. Note methodological limitations
6. Be objective in interpretation
7. CRITICAL: Ensure all tables use proper markdown syntax with pipes (|) and dashes (-)

EXAMPLE TABLE FORMAT:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

CONTEXT:
{context}""",
                ),
                ("human", "{question}"),
            ]
        )

        # General comparison (flexible)
        self.general_comparison_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical research expert performing comparative analysis.

TASK: Compare the specified items systematically.

REQUIRED STRUCTURE:

## Comparison Overview
[Introduction to what is being compared]

## Side-by-Side Comparison

| Aspect | Item A | Item B | Analysis |
|--------|--------|--------|----------|
| [Aspect 1] | [Details] [citation] | [Details] [citation] | [Comparison] |
| [Aspect 2] | [Details] [citation] | [Details] [citation] | [Comparison] |
| [Aspect 3] | [Details] [citation] | [Details] [citation] | [Comparison] |

## Similarities
[List common features or findings]

## Key Differences
[Highlight important distinctions]

## Advantages and Disadvantages

**Item A:**
- Advantages: [List with citations]
- Disadvantages: [List with citations]

**Item B:**
- Advantages: [List with citations]
- Disadvantages: [List with citations]

## Recommendations
[Evidence-based guidance on selection or interpretation]

INSTRUCTIONS:
1. Use ONLY information from the provided context
2. Use MARKDOWN TABLES for structured comparison with proper formatting (use | for columns, separate header with |---|---|)
3. Include specific data and citations
4. Be balanced and objective
5. Note when information is limited
6. CRITICAL: Ensure all tables use proper markdown syntax with pipes (|) and dashes (-)

EXAMPLE TABLE FORMAT:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

CONTEXT:
{context}""",
                ),
                ("human", "{question}"),
            ]
        )

        logger.info("Initialized comparison prompts")

    def detect_comparison_type(self, query: str) -> str:
        """
        Detect what type of comparison is being requested

        Returns: 'drug', 'study', or 'general'
        """
        query_lower = query.lower()

        # Drug/treatment comparison indicators
        drug_keywords = [
            "drug",
            "medication",
            "treatment",
            "therapy",
            "agent",
            "pembrolizumab",
            "nivolumab",
            "chemotherapy",
            "immunotherapy",
            "efficacy",
            "safety",
            "adverse event",
            "side effect",
        ]

        # Study comparison indicators
        study_keywords = [
            "study",
            "trial",
            "research",
            "investigation",
            "phase",
            "nct",
            "pmid",
            "clinical trial",
        ]

        drug_score = sum(1 for kw in drug_keywords if kw in query_lower)
        study_score = sum(1 for kw in study_keywords if kw in query_lower)

        if drug_score > study_score:
            return "drug"
        elif study_score > drug_score:
            return "study"
        else:
            return "general"

    def extract_comparison_items(self, query: str) -> tuple:
        """
        Extract the two items being compared from the query

        Returns: (item_a, item_b) or (None, None)
        """
        import re

        # Pattern: "X vs Y" or "X versus Y" or "compare X and Y"
        patterns = [
            r"(\w+(?:\s+\w+)*)\s+(?:vs\.?|versus)\s+(\w+(?:\s+\w+)*)",
            r"compare\s+(\w+(?:\s+\w+)*)\s+(?:and|with|to)\s+(\w+(?:\s+\w+)*)",
            r"difference\s+between\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()

        return None, None

    def _ensure_markdown_tables(self, text: str) -> str:
        """
        Post-process text to ensure tables are in proper markdown format

        Args:
            text: Generated text that may contain malformed tables

        Returns:
            Text with properly formatted markdown tables
        """
        import re

        lines = text.split("\n")
        processed_lines = []
        in_table = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect table rows (lines with multiple | characters)
            if "|" in stripped and stripped.count("|") >= 2:
                # Clean up the line
                parts = [p.strip() for p in stripped.split("|")]
                # Remove empty first/last elements if line starts/ends with |
                if parts and parts[0] == "":
                    parts = parts[1:]
                if parts and parts[-1] == "":
                    parts = parts[:-1]

                if parts:
                    # Reconstruct as proper markdown table row
                    formatted_line = "| " + " | ".join(parts) + " |"

                    # If this is the first row of a table, add separator after it
                    if not in_table:
                        processed_lines.append(formatted_line)
                        # Add separator row if next line isn't already a separator
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if not (next_line.startswith("|") and "-" in next_line):
                                # Create separator with correct number of columns
                                separator = (
                                    "| " + " | ".join(["---"] * len(parts)) + " |"
                                )
                                processed_lines.append(separator)
                        in_table = True
                    else:
                        # Check if this is a separator row
                        if all(
                            p.replace("-", "").replace(" ", "") == "" for p in parts
                        ):
                            # It's a separator, ensure proper format
                            separator = "| " + " | ".join(["---"] * len(parts)) + " |"
                            processed_lines.append(separator)
                        else:
                            # Regular data row
                            processed_lines.append(formatted_line)
                else:
                    processed_lines.append(line)
                    in_table = False
            else:
                # Not a table row
                processed_lines.append(line)
                in_table = False

        return "\n".join(processed_lines)

    def compare(
        self, query: str, context: str, comparison_type: Optional[str] = None
    ) -> str:
        """
        Generate a systematic comparison

        Args:
            query: Comparison query
            context: Formatted context documents
            comparison_type: Type of comparison ('drug', 'study', 'general')

        Returns:
            Structured comparison text
        """
        try:
            # Auto-detect comparison type if not provided
            if comparison_type is None:
                comparison_type = self.detect_comparison_type(query)

            # Select appropriate prompt (no dynamic replacement needed)
            if comparison_type == "drug":
                prompt = self.drug_comparison_prompt
            elif comparison_type == "study":
                prompt = self.study_comparison_prompt
            else:
                prompt = self.general_comparison_prompt

            # Build chain
            chain = (
                {"context": lambda _: context, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Generate comparison
            result = chain.invoke(query)

            # Post-process to ensure proper markdown table formatting
            result = self._ensure_markdown_tables(result)

            logger.info(
                f"Generated {comparison_type} comparison with proper markdown tables"
            )
            return result

        except Exception as e:
            logger.error(f"Comparison generation error: {e}")
            return f"Error generating comparison: {str(e)}"


def get_comparison_chain(llm) -> ComparisonChain:
    """Create a comparison chain instance"""
    return ComparisonChain(llm)
