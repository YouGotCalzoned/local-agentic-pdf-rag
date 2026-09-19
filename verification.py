import re


def invoke_llm(llm, prompt):
    """
    Invoke the LLM and safely return plain text.
    """

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return response.content.strip()

    return str(response).strip()


def extract_source_ids(text):
    """
    Extract source IDs from text.

    Examples:
        [S1]
        [S1, S3]
        S2, S7
        (S4)

    Returns:
        ["S1", "S3"]
    """

    source_ids = re.findall(
        r"\bS\d+\b",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize to uppercase and preserve order.
    unique_sources = []

    for source_id in source_ids:
        source_id = source_id.upper()

        if source_id not in unique_sources:
            unique_sources.append(source_id)

    return unique_sources


def remove_source_ids(text):
    """
    Remove citation markers from a claim.

    Example:

        "Key-value stores use primary-key access. [S6]"

    becomes:

        "Key-value stores use primary-key access."
    """

    cleaned = re.sub(
        r"\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\(\s*S\d+(?:\s*,\s*S\d+)*\s*\)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned.strip()


def extract_claims(llm, draft_answer):
    """
    Break the draft answer into atomic factual claims.

    IMPORTANT:
    Each claim must preserve the citation(s) from the draft.

    Example output:

        Key-value stores use primary-key access. [S6]
        Document databases store self-describing documents. [S11]

    Claims without citations will later be rejected
    deterministically by Python.
    """

    prompt = f"""
You are extracting factual claims from a draft RAG answer.

Break the answer into small, atomic factual claims.

RULES:

1. Each claim should contain only ONE main factual statement.

2. PRESERVE the source citation attached to the statement.

3. Citations must use this exact format:

   [S1]
   [S2, S4]

4. Do NOT invent citations.

5. Do NOT convert a factual claim into a statement about the source.

BAD:
"The passage in S6 mentions a key-value database."

GOOD:
"A key-value store is a simple hash table. [S6]"

6. If a statement has no citation in the draft, output it
   without inventing one.

7. Do not number the claims.

Return ONLY the claims, one claim per line.

DRAFT ANSWER:

{draft_answer}
"""

    response_text = invoke_llm(
        llm,
        prompt,
    )

    claims = []

    for line in response_text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove accidental bullets / numbering.
        line = re.sub(
            r"^\s*[-*]\s*",
            "",
            line,
        )

        line = re.sub(
            r"^\s*\d+[\.\)]\s*",
            "",
            line,
        )

        if line:
            claims.append(line)

    return claims


def parse_labeled_context(labeled_context):
    """
    Convert labeled context into:

        {
            "S1": "chunk text...",
            "S2": "chunk text...",
        }

    This allows Python to resolve citations deterministically.

    Expected context structure is roughly:

        [S1]
        chunk text

        [S2]
        chunk text

    The parser is intentionally tolerant of formatting differences.
    """

    source_map = {}

    pattern = re.compile(
        r"\[(S\d+)\]\s*(.*?)(?=\n\s*\[S\d+\]|\Z)",
        flags=re.DOTALL | re.IGNORECASE,
    )

    matches = pattern.findall(
        labeled_context
    )

    for source_id, content in matches:

        source_id = source_id.upper()

        source_map[source_id] = (
            content.strip()
        )

    return source_map


def build_cited_evidence(
    source_ids,
    source_map,
):
    """
    Build a context containing ONLY the chunks cited
    by the claim.
    """

    evidence_parts = []

    for source_id in source_ids:

        content = source_map.get(
            source_id
        )

        if content:

            evidence_parts.append(
                f"[{source_id}]\n{content}"
            )

    return "\n\n".join(
        evidence_parts
    )


def verify_single_claim(
    llm,
    claim_text,
    cited_evidence,
):
    """
    Verify one factual claim against ONLY its cited evidence.

    The verifier is NOT asked to search the entire corpus.
    It performs a narrow textual entailment check.
    """

    prompt = f"""
You are a strict citation verifier for a RAG system.

Your task is very narrow.

Determine whether the CLAIM is supported by the CITED EVIDENCE.

CLAIM:

{claim_text}


CITED EVIDENCE:

{cited_evidence}


RULES:

1. Judge ONLY using the cited evidence above.

2. Do NOT use outside knowledge.

3. The wording does not need to match exactly.
   Semantic equivalence is acceptable.

4. Minor paraphrasing is acceptable.

5. If the evidence clearly entails the claim, return:

SUPPORTED

6. If the evidence does not clearly support the claim, return:

UNSUPPORTED

7. Do not explain your reasoning.

Return exactly ONE word:

SUPPORTED

or

UNSUPPORTED
"""

    result = invoke_llm(
        llm,
        prompt,
    )

    result_upper = (
        result
        .strip()
        .upper()
    )

    # Deterministic parsing.
    if result_upper.startswith(
        "SUPPORTED"
    ):
        return True, result

    return False, result


def verify_claims(
    llm,
    claims,
    labeled_context,
):
    """
    Citation-scoped claim verification.

    Pipeline:

        claim
          ↓
        extract [S#]
          ↓
        Python resolves exact chunks
          ↓
        verifier sees ONLY those chunks
          ↓
        SUPPORTED / UNSUPPORTED
          ↓
        Python keeps or discards claim

    Returns:

        verified_claims
        debug_results

    This preserves the interface expected by rag.py.
    """

    source_map = parse_labeled_context(
        labeled_context
    )

    verified_claims = []

    debug_results = []

    for index, original_claim in enumerate(
        claims,
        start=1,
    ):

        source_ids = extract_source_ids(
            original_claim
        )

        claim_text = remove_source_ids(
            original_claim
        )

        # -----------------------------------------------------
        # Rule 1:
        # No citation = deterministic rejection.
        # -----------------------------------------------------

        if not source_ids:

            debug_results.append(
                {
                    "index": index,
                    "claim": claim_text,
                    "verification_text": (
                        "UNSUPPORTED - no citation provided"
                    ),
                    "kept": False,
                }
            )

            continue

        # -----------------------------------------------------
        # Rule 2:
        # Resolve cited evidence using Python.
        # -----------------------------------------------------

        cited_evidence = (
            build_cited_evidence(
                source_ids,
                source_map,
            )
        )

        if not cited_evidence:

            debug_results.append(
                {
                    "index": index,
                    "claim": claim_text,
                    "verification_text": (
                        "UNSUPPORTED - cited source "
                        "could not be resolved"
                    ),
                    "kept": False,
                }
            )

            continue

        # -----------------------------------------------------
        # Rule 3:
        # LLM performs only the entailment judgment.
        # -----------------------------------------------------

        is_supported, verification_text = (
            verify_single_claim(
                llm,
                claim_text,
                cited_evidence,
            )
        )

        debug_results.append(
            {
                "index": index,
                "claim": claim_text,
                "verification_text": (
                    verification_text
                ),
                "kept": is_supported,
            }
        )

        if is_supported:

            verified_claims.append(
                {
                    "claim": claim_text,
                    "sources": ", ".join(
                        source_ids
                    ),
                }
            )

    return (
        verified_claims,
        debug_results,
    )