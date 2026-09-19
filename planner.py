import re

from config import (
    MAX_EVIDENCE_REQUIREMENTS,
    MAX_FOLLOW_UP_QUERIES,
)


# ============================================================
# BASIC LLM HELPER
# ============================================================

def invoke_llm(llm, prompt):
    """
    Invoke the LLM and return plain stripped text.

    Keeping this in one place makes planner functions easier
    to read.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def normalize_query(query):
    """
    Normalize a search query so exact / near-format duplicate
    searches can be detected deterministically.

    This is NOT semantic duplicate detection.

    Example:

        "CAP theorem availability"
        "  CAP theorem availability  "

    become identical.

    But:

        "CAP theorem availability"
        "availability tradeoff under network partitions"

    remain different.
    """

    query = query.lower().strip()

    query = re.sub(
        r"\s+",
        " ",
        query,
    )

    return query


# ============================================================
# EVIDENCE REQUIREMENT DERIVATION
# ============================================================

def derive_evidence_requirements(
    llm,
    question,
):
    """
    Break the ORIGINAL user question into a small number of
    evidence requirements.

    These requirements represent WHAT MUST BE SUPPORTED
    by retrieved evidence for the question to be answered well.

    Example:

        Question:
        Explain how replication, consistency, and availability
        are related.

        Possible requirements:

        R1: Explain what replication does.
        R2: Explain how replication affects consistency.
        R3: Explain how replication affects availability.
        R4: Explain consistency vs availability tradeoffs.
        R5: Explain the role of network partitions / CAP.

    IMPORTANT:

    The model is NOT answering the question here.

    It is defining the evidence plan.
    """

    prompt = f"""
You are planning evidence requirements for a Retrieval-Augmented
Generation system.

DO NOT answer the user's question.

Your task is to determine what evidence would be required from
the knowledge base to answer the ORIGINAL USER QUESTION fully.

ORIGINAL USER QUESTION:

{question}

Break the question into the smallest useful set of evidence
requirements.

Rules:

- Produce at most {MAX_EVIDENCE_REQUIREMENTS} requirements.
- Requirements must come from the user's actual question.
- Do not introduce unrelated topics.
- Do not require unnecessary detail.
- A simple question may need only one requirement.
- A multi-part or relationship question may require several.
- Each requirement should describe evidence that could be
  supported by a passage from the source.
- Do not answer the question.
- Do not generate search queries yet.

Return EXACTLY this format:

R1: <requirement>
R2: <requirement>
R3: <requirement>

Continue only as far as needed.
"""

    response_text = invoke_llm(
        llm,
        prompt,
    )

    requirements = []

    for line in response_text.splitlines():

        line = line.strip()

        match = re.match(
            r"^R(\d+)\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        requirement_number = int(
            match.group(1)
        )

        requirement_text = (
            match.group(2).strip()
        )

        if not requirement_text:
            continue

        requirements.append(
            {
                "id": f"R{requirement_number}",
                "requirement": requirement_text,
            }
        )

    # --------------------------------------------------------
    # Defensive fallback
    #
    # If the LLM ignores the requested format completely,
    # the system should still work.
    # --------------------------------------------------------

    if not requirements:

        requirements = [
            {
                "id": "R1",
                "requirement": question,
            }
        ]

    # --------------------------------------------------------
    # Enforce deterministic maximum.
    # --------------------------------------------------------

    requirements = requirements[
        :MAX_EVIDENCE_REQUIREMENTS
    ]

    # --------------------------------------------------------
    # Re-number requirements.
    #
    # This prevents weird model output such as:
    #
    # R1
    # R3
    # R7
    #
    # from leaking into application state.
    # --------------------------------------------------------

    normalized_requirements = []

    for index, item in enumerate(
        requirements,
        start=1,
    ):

        normalized_requirements.append(
            {
                "id": f"R{index}",
                "requirement": (
                    item["requirement"]
                ),
            }
        )

    return normalized_requirements


# ============================================================
# REQUIREMENT FORMATTING
# ============================================================

def format_requirements(requirements):
    """
    Convert requirement objects into readable prompt text.
    """

    lines = []

    for item in requirements:

        lines.append(
            f"{item['id']}: "
            f"{item['requirement']}"
        )

    return "\n".join(lines)


# ============================================================
# COVERAGE EVALUATION
# ============================================================

def evaluate_requirement_coverage(
    llm,
    question,
    requirements,
    context,
):
    """
    Evaluate every evidence requirement against the CURRENT
    accumulated retrieved context.

    Each requirement receives one status:

        COVERED
        PARTIAL
        MISSING

    COVERED:
        Direct evidence exists.

    PARTIAL:
        Some relevant evidence exists, but it is not enough
        to support the complete requirement.

    MISSING:
        The current context does not provide useful direct
        evidence for that requirement.

    The LLM makes the semantic judgment.

    Python owns the requirement IDs and state.
    """

    requirements_text = (
        format_requirements(
            requirements
        )
    )

    prompt = f"""
You are evaluating evidence coverage in a
Retrieval-Augmented Generation system.

DO NOT answer the user's question.

ORIGINAL USER QUESTION:

{question}


EVIDENCE REQUIREMENTS:

{requirements_text}


CURRENT RETRIEVED CONTEXT:

{context}


For EACH evidence requirement, determine whether the current
retrieved context contains enough DIRECT evidence.

Allowed statuses:

COVERED
- Direct evidence exists that can support the requirement.

PARTIAL
- Some relevant evidence exists, but important information
  needed by the requirement is still missing.

MISSING
- The retrieved evidence does not directly support the
  requirement.

Important rules:

- Judge ONLY from CURRENT RETRIEVED CONTEXT.
- Do not use your pretrained knowledge.
- Do not assume facts that are not present.
- Do not answer the original question.
- Do not invent new requirements.
- Preserve the requirement IDs exactly.
- A keyword appearing somewhere is not automatically enough
  to mark a requirement COVERED.
- Prefer PARTIAL over COVERED when evidence is incomplete.

Return EXACTLY one line per requirement:

R1|COVERED|short reason
R2|PARTIAL|short reason
R3|MISSING|short reason
"""

    response_text = invoke_llm(
        llm,
        prompt,
    )

    parsed_statuses = {}

    for line in response_text.splitlines():

        line = line.strip()

        parts = line.split(
            "|",
            2,
        )

        if len(parts) < 2:
            continue

        requirement_id = (
            parts[0]
            .strip()
            .upper()
        )

        status = (
            parts[1]
            .strip()
            .upper()
        )

        reason = ""

        if len(parts) == 3:
            reason = parts[2].strip()

        if status not in {
            "COVERED",
            "PARTIAL",
            "MISSING",
        }:
            continue

        parsed_statuses[
            requirement_id
        ] = {
            "status": status,
            "reason": reason,
        }

    # --------------------------------------------------------
    # Build deterministic state using OUR requirement list.
    #
    # The LLM cannot add/remove requirement IDs.
    # --------------------------------------------------------

    coverage_state = []

    for requirement in requirements:

        requirement_id = (
            requirement["id"]
        )

        parsed = parsed_statuses.get(
            requirement_id,
            {
                "status": "MISSING",
                "reason": (
                    "Planner did not return "
                    "a valid coverage decision."
                ),
            },
        )

        coverage_state.append(
            {
                "id": requirement_id,
                "requirement": (
                    requirement[
                        "requirement"
                    ]
                ),
                "status": (
                    parsed["status"]
                ),
                "reason": (
                    parsed["reason"]
                ),
            }
        )

    return coverage_state


# ============================================================
# COVERAGE HELPERS
# ============================================================

def all_requirements_covered(
    coverage_state,
):
    """
    Deterministically decide whether retrieval may stop.

    We do NOT ask the LLM:

        "Should we stop?"

    Python simply checks whether every requirement has the
    status COVERED.
    """

    if not coverage_state:
        return False

    return all(
        item["status"] == "COVERED"
        for item in coverage_state
    )


def get_uncovered_requirements(
    coverage_state,
):
    """
    Return PARTIAL and MISSING requirements.

    These are the goals for the next search round.
    """

    return [
        item
        for item in coverage_state
        if item["status"]
        != "COVERED"
    ]


# ============================================================
# FOLLOW-UP SEARCH PLANNING
# ============================================================

def plan_follow_up_searches(
    llm,
    question,
    coverage_state,
    context,
    previous_searches=None,
):
    """
    Generate searches specifically for requirements that are
    currently PARTIAL or MISSING.

    This replaces the old vague pattern:

        "Something is missing. Search for more."

    with:

        R3 MISSING
            ↓
        generate search specifically for R3
    """

    if previous_searches is None:
        previous_searches = []

    uncovered = (
        get_uncovered_requirements(
            coverage_state
        )
    )

    if not uncovered:
        return []

    uncovered_lines = []

    for item in uncovered:

        uncovered_lines.append(
            f"{item['id']} | "
            f"{item['status']} | "
            f"{item['requirement']}"
        )

    uncovered_text = "\n".join(
        uncovered_lines
    )

    if previous_searches:

        previous_search_text = "\n".join(
            f"- {query}"
            for query in previous_searches
        )

    else:

        previous_search_text = (
            "(none)"
        )

    prompt = f"""
You are planning follow-up vector searches for a
Retrieval-Augmented Generation system.

DO NOT answer the user's question.

ORIGINAL USER QUESTION:

{question}


REQUIREMENTS THAT ARE STILL PARTIAL OR MISSING:

{uncovered_text}


SEARCHES ALREADY PERFORMED:

{previous_search_text}


CURRENT RETRIEVED CONTEXT:

{context}


Generate focused semantic-search queries that are likely to
retrieve evidence for the PARTIAL or MISSING requirements.

Rules:

- Generate at most {MAX_FOLLOW_UP_QUERIES} searches.
- Prioritize MISSING requirements before PARTIAL ones.
- Search only for information needed by the requirements.
- Do not repeat searches already performed.
- Do not search for requirements already marked COVERED.
- Prefer specific searches over generic searches.
- Use terminology suggested by the retrieved source when useful.
- Do not answer the user's question.
- Do not explain your choices.

Return EXACTLY:

SEARCH:
<query>
<query>
<query>
"""

    response_text = invoke_llm(
        llm,
        prompt,
    )

    return parse_follow_up_searches(
        response_text,
        previous_searches,
    )


# ============================================================
# SEARCH QUERY PARSING
# ============================================================

def parse_follow_up_searches(
    planner_text,
    previous_searches=None,
):
    """
    Parse planner-generated searches.

    Python handles:

    - cleanup
    - exact normalized duplicate removal
    - previous-search filtering
    - maximum query count
    """

    if previous_searches is None:
        previous_searches = []

    if "SEARCH:" not in planner_text.upper():
        return []

    match = re.search(
        r"SEARCH:\s*(.*)",
        planner_text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if not match:
        return []

    search_section = match.group(1)

    previous_normalized = {
        normalize_query(query)
        for query in previous_searches
    }

    new_queries = []

    seen_new_queries = set()

    for line in search_section.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove common bullet / numbering formats.
        line = re.sub(
            r"^[\-\*\•\d\.\)\s]+",
            "",
            line,
        ).strip()

        if not line:
            continue

        normalized = normalize_query(
            line
        )

        if not normalized:
            continue

        # Defensive guard against the LLM accidentally returning
        # another SEARCH header inside the search section.
        if normalized in {
            "search",
            "search:",
        }:
            continue
        
        # Already searched in a previous round.
        if (
            normalized
            in previous_normalized
        ):
            continue

        # Duplicate within this planner response.
        if (
            normalized
            in seen_new_queries
        ):
            continue

        seen_new_queries.add(
            normalized
        )

        new_queries.append(
            line
        )

        if (
            len(new_queries)
            >= MAX_FOLLOW_UP_QUERIES
        ):
            break

    return new_queries