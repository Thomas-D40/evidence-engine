"""
Common prompt components and instructions.

Reusable prompt snippets that can be composed into full prompts.
"""

# ============================================================================
# JSON OUTPUT INSTRUCTIONS
# ============================================================================

JSON_OUTPUT_STRICT = """
**CRITICAL: OUTPUT FORMAT**
- Return ONLY valid JSON, no explanations before or after
- No markdown code blocks (no ```json```)
- Ensure proper JSON escaping for quotes and special characters
- Structure must exactly match the schema provided
"""

JSON_OUTPUT_RELAXED = """
**OUTPUT FORMAT**
Return valid JSON matching the schema.
If uncertain about a field, use null or an empty value of the correct type.
"""

# ============================================================================
# COMMON ANALYSIS PRINCIPLES
# ============================================================================

OBJECTIVITY_INSTRUCTION = """
**OBJECTIVITY REQUIREMENT**
- Remain strictly neutral and factual
- Do not inject personal opinions or biases
- Extract only what is explicitly stated or clearly implied
- Distinguish between facts and opinions in the content
"""

COMPREHENSIVENESS_INSTRUCTION = """
**COMPREHENSIVENESS**
- Aim for thorough coverage of the content
- Don't limit yourself to only the most obvious points
- Include nuanced or conditional statements
- Balance between completeness and relevance
"""

# ============================================================================
# EVIDENCE HANDLING
# ============================================================================

CITATION_INSTRUCTION = """
**CITATION REQUIREMENT**
- Every claim must reference its source
- Use direct quotes or close paraphrasing
- Maintain accuracy and context of original statement
"""

SOURCE_QUALITY_INSTRUCTION = """
**SOURCE QUALITY ASSESSMENT**
- Consider the reliability and authority of sources
- Note the publication type (peer-reviewed, blog, etc.)
- Identify potential biases or conflicts of interest
"""

# ============================================================================
# COMMON EXCLUSIONS
# ============================================================================

EXCLUSION_CRITERIA_ARGUMENTS = """
**WHAT TO EXCLUDE:**
- Simple anecdotes or personal stories without broader claims
- Rhetorical questions without clear stance
- Purely descriptive statements
- Transition phrases or filler content
- Examples that only illustrate a point (include the point, not the example)
"""

# ============================================================================
# STANCE CLASSIFICATION
# ============================================================================

STANCE_DEFINITIONS = """
**STANCE CLASSIFICATION:**
- **affirmatif**: The author presents this as fact or strongly advocates for it
- **conditionnel**: The author presents this hypothetically, conditionally, or with uncertainty
"""

# ============================================================================
# JSON SCHEMAS (Common Structures)
# ============================================================================

SCHEMA_ARGUMENT = {
    "type": "object",
    "properties": {
        "argument": {"type": "string", "description": "The main claim or thesis"},
        "stance": {"type": "string", "enum": ["affirmatif", "conditionnel"]},
    },
    "required": ["argument", "stance"]
}

SCHEMA_SOURCE_REFERENCE = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "source": {"type": "string", "description": "URL of the source"}
    },
    "required": ["claim", "source"]
}

SCHEMA_PROS_CONS = {
    "type": "object",
    "properties": {
        "pros": {
            "type": "array",
            "items": SCHEMA_SOURCE_REFERENCE,
            "description": "Points that support or corroborate the argument"
        },
        "cons": {
            "type": "array",
            "items": SCHEMA_SOURCE_REFERENCE,
            "description": "Points that contradict or challenge the argument"
        }
    }
}
