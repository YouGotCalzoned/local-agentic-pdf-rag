import re

from context import build_labeled_context


# ============================================================
# BASIC LLM HELPER
# ============================================================

def invoke_llm(llm, prompt):
    """
    Invoke the LLM and return stripped plain text.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


# ============================================================
# SOURCE-ID HELPERS
# ============================================================

def extract_source_ids(text):
    """
    Extract source IDs such as:

        S1
        S2
        S10

    from model output.

    Python owns source resolution.

    The LLM may say:

        SUPPORT: S2, S5

    but it never directly controls Document objects.
    """

    if not text:
        return []

    source_ids = re.findall(
        r"\bS\d+\b",
        text,
        flags=re.IGNORECASE,
    )

    normalized_ids = []

    seen = set()

    for source_id in source_ids:

        source_id = (
            source_id.upper()
        )

        if source_id in seen:
            continue

        seen.add(source_id)

        normalized_ids.append(
            source_id
        )

    return normalized_ids


# ============================================================
# SOURCE MAP
# ============================================================

def build_document_source_map(results):
    """
    Create a deterministic mapping:

        S1 -> Document 1
        S2 -> Document 2
        S3 -> Document 3

    IMPORTANT:

    This namespace exists ONLY during evidence attribution.

    It is NOT necessarily the same S1...Sn namespace later
    shown to the answer generator.

    After evidence selection, context.py will create a fresh
    final generation namespace.
    """

    source_map = {}

    for index, document in enumerate(
        results,
        start=1,
    ):

        source_id = f"S{index}"

        source_map[source_id] = (
            document
        )

    return source_map


# ============================================================
# SUPPORT JUDGMENT
# ============================================================

def evaluate_requirement_support(
    llm,
    question,
    requirement,
    labeled_context,
):
    """
    Ask the LLM which retrieved passages DIRECTLY SUPPORT
    one evidence requirement.

    This is deliberately different from semantic similarity.

    Embeddings answer:

        "Does this passage resemble the requirement?"

    This function asks:

        "Does this passage actually provide evidence that
         helps satisfy the requirement?"

    Multiple passages may jointly support one requirement.
    """

    requirement_id = (
        requirement["id"]
    )

    requirement_text = (
        requirement["requirement"]
    )

    prompt = f"""
You are performing evidence attribution for a
Retrieval-Augmented Generation system.

DO NOT answer the user's question.

Your task is to identify which source passages actually provide
DIRECT evidence for ONE evidence requirement.


ORIGINAL USER QUESTION:

{question}


EVIDENCE REQUIREMENT:

{requirement_id}: {requirement_text}


RETRIEVED SOURCES:

{labeled_context}


Determine which source passages directly help satisfy the
evidence requirement.

Important rules:

- Judge ONLY from the retrieved sources above.
- Do not use pretrained knowledge.
- Semantic similarity alone is NOT enough.
- A source must contain useful explanatory or factual evidence.
- An index, table of contents, bibliography, glossary-like list,
  or keyword list should NOT count merely because it contains
  relevant words.
- A source does NOT need to answer the entire requirement alone.
- Multiple sources may jointly satisfy the requirement.
- Select a source if it contributes an important piece of
  evidence needed for the requirement.
- Do not invent source IDs.
- Do not explain the original question.
- Be conservative, but do not require one passage to contain
  the complete synthesized answer.

Return EXACTLY one of these forms:

SUPPORT: S2, S5, S7

or:

SUPPORT: NONE
"""

    response_text = invoke_llm(
        llm,
        prompt,
    )

    # --------------------------------------------------------
    # Explicit NONE response.
    # --------------------------------------------------------

    if re.search(
        r"SUPPORT\s*:\s*NONE",
        response_text,
        flags=re.IGNORECASE,
    ):
        return []

    # --------------------------------------------------------
    # Extract whatever S IDs the model selected.
    # --------------------------------------------------------

    source_ids = (
        extract_source_ids(
            response_text
        )
    )

    return source_ids


# ============================================================
# REQUIREMENT -> DOCUMENT ATTRIBUTION
# ============================================================

def attribute_evidence_to_requirements(
    llm,
    question,
    requirements,
    results,
):
    """
    Build an evidence-support map.

    Example:

        {
            "R1": {
                "requirement": "...",
                "source_ids": ["S1", "S4"],
                "documents": [doc1, doc4],
            },

            "R2": {
                "requirement": "...",
                "source_ids": ["S7", "S9"],
                "documents": [doc7, doc9],
            }
        }

    Architecture:

        requirements
             +
        retrieved chunks
             ↓
        LLM support judgment
             ↓
        source IDs
             ↓
        Python source resolution
             ↓
        requirement -> actual Documents
    """

    if not requirements:
        return {}

    if not results:
        return {}

    # --------------------------------------------------------
    # Build ONE attribution namespace over the entire current
    # candidate pool.
    #
    # Example:
    #
    # S1 = page 11
    # S2 = page 52
    # S3 = page 145
    # ...
    # --------------------------------------------------------

    labeled_context = (
        build_labeled_context(
            results
        )
    )

    source_map = (
        build_document_source_map(
            results
        )
    )

    attribution_map = {}

    print(
        "\n"
        "=========================================="
    )

    print(
        "EVIDENCE ATTRIBUTION"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Evaluate requirements independently.
    #
    # This is important.
    #
    # We do NOT ask the model to solve all requirements in one
    # giant output because that makes parsing and attribution
    # less reliable.
    # --------------------------------------------------------

    for requirement in requirements:

        requirement_id = (
            requirement["id"]
        )

        requirement_text = (
            requirement[
                "requirement"
            ]
        )

        source_ids = (
            evaluate_requirement_support(
                llm,
                question,
                requirement,
                labeled_context,
            )
        )

        # ----------------------------------------------------
        # Python rejects hallucinated / invalid source IDs.
        # ----------------------------------------------------

        valid_source_ids = [
            source_id
            for source_id in source_ids
            if source_id in source_map
        ]

        documents = [
            source_map[source_id]
            for source_id
            in valid_source_ids
        ]

        attribution_map[
            requirement_id
        ] = {
            "id": requirement_id,
            "requirement": (
                requirement_text
            ),
            "source_ids": (
                valid_source_ids
            ),
            "documents": documents,
        }

        # ----------------------------------------------------
        # Diagnostics.
        # ----------------------------------------------------

        print()
        print(
            f"{requirement_id}: "
            f"{requirement_text}"
        )

        if not valid_source_ids:

            print(
                "Supporting evidence: NONE ❌"
            )

            continue

        print(
            "Supporting evidence:"
        )

        for source_id in (
            valid_source_ids
        ):

            document = (
                source_map[source_id]
            )

            page = (
                document.metadata.get(
                    "page",
                    "unknown",
                )
            )

            preview = (
                document.page_content
                .replace("\n", " ")
                .strip()
            )

            preview = preview[:220]

            print(
                f"- {source_id} | "
                f"Page {page}"
            )

            print(
                f"  {preview}"
            )

        print(
            "Decision: "
            "REQUIREMENT HAS SUPPORT ✅"
        )

    return attribution_map


# ============================================================
# ATTRIBUTION COVERAGE STATE
# ============================================================

def build_attribution_coverage_state(
    requirements,
    attribution_map,
):
    """
    Build a simple deterministic coverage state from the
    attribution map.

    IMPORTANT:

    This is intentionally simpler than planner.py's semantic:

        COVERED
        PARTIAL
        MISSING

    Here we only ask:

        "Did we find at least one directly supporting passage?"

    So the state is:

        SUPPORTED
        UNSUPPORTED

    We should NOT pretend that one supporting passage proves
    that an entire complex requirement is fully covered.
    """

    coverage_state = []

    for requirement in requirements:

        requirement_id = (
            requirement["id"]
        )

        attribution = (
            attribution_map.get(
                requirement_id,
                {},
            )
        )

        source_ids = (
            attribution.get(
                "source_ids",
                [],
            )
        )

        if source_ids:

            status = "SUPPORTED"

        else:

            status = "UNSUPPORTED"

        coverage_state.append(
            {
                "id": requirement_id,
                "requirement": (
                    requirement[
                        "requirement"
                    ]
                ),
                "status": status,
                "source_ids": (
                    source_ids
                ),
            }
        )

    return coverage_state


# ============================================================
# UNIQUE SUPPORTING DOCUMENTS
# ============================================================

def collect_supporting_documents(
    requirements,
    attribution_map,
):
    """
    Collect supporting Documents in requirement order while
    removing duplicates.

    Example:

        R1 -> doc A, doc B
        R2 -> doc B, doc C
        R3 -> doc D

    becomes:

        A, B, C, D

    This function does NOT apply the final context budget.

    evidence.py will own that responsibility.
    """

    selected_documents = []

    seen_document_keys = set()

    for requirement in requirements:

        requirement_id = (
            requirement["id"]
        )

        attribution = (
            attribution_map.get(
                requirement_id,
                {},
            )
        )

        documents = (
            attribution.get(
                "documents",
                [],
            )
        )

        for document in documents:

            # ------------------------------------------------
            # Build a deterministic identity key.
            #
            # Metadata alone may not always uniquely identify
            # chunks from the same page, so include content.
            # ------------------------------------------------

            source = (
                document.metadata.get(
                    "source",
                    ""
                )
            )

            page = (
                document.metadata.get(
                    "page",
                    ""
                )
            )

            document_key = (
                source,
                page,
                document.page_content,
            )

            if (
                document_key
                in seen_document_keys
            ):
                continue

            seen_document_keys.add(
                document_key
            )

            selected_documents.append(
                document
            )

    return selected_documents


# ============================================================
# DEBUG SUMMARY
# ============================================================

def print_attribution_summary(
    requirements,
    attribution_map,
):
    """
    Print a compact requirement -> source summary.
    """

    print(
        "\n"
        "------------------------------------------"
    )

    print(
        "EVIDENCE ATTRIBUTION SUMMARY"
    )

    print(
        "------------------------------------------"
    )

    for requirement in requirements:

        requirement_id = (
            requirement["id"]
        )

        attribution = (
            attribution_map.get(
                requirement_id,
                {},
            )
        )

        source_ids = (
            attribution.get(
                "source_ids",
                [],
            )
        )

        if source_ids:

            sources_text = ", ".join(
                source_ids
            )

        else:

            sources_text = "NONE"

        print(
            f"{requirement_id} "
            f"-> {sources_text}"
        )