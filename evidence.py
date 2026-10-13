import math

from config import (
    MAX_GENERATION_CHUNKS,
    MAX_GENERATION_CONTEXT_CHARS,
    GENERATION_MMR_LAMBDA,
)


# ============================================================
# VECTOR HELPERS
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b,
):
    """
    Calculate cosine similarity between two vectors.
    """

    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )

    norm_a = math.sqrt(
        sum(
            a * a
            for a in vector_a
        )
    )

    norm_b = math.sqrt(
        sum(
            b * b
            for b in vector_b
        )
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return (
        dot_product
        / (norm_a * norm_b)
    )


# ============================================================
# DEBUG HELPERS
# ============================================================

def get_page(document):
    """
    Safely retrieve page metadata.
    """

    return document.metadata.get(
        "page",
        "unknown",
    )


def get_preview(
    document,
    length=220,
):
    """
    Produce a short one-line chunk preview.
    """

    text = (
        document.page_content
        .replace("\n", " ")
        .strip()
    )

    return text[:length]


# ============================================================
# DOCUMENT IDENTITY
# ============================================================

def get_document_key(document):
    """
    Build a deterministic identity for a chunk.

    A chunk is uniquely identified by:

        document_id + chunk_id
    """

    return (
        document.metadata.get(
            "document_id",
            "",
        ),
        document.metadata.get(
            "chunk_id",
            "",
        ),
    )


# ============================================================
# CANDIDATE PREPARATION
# ============================================================

def prepare_candidates(
    embeddings,
    question,
    results,
):
    """
    Embed the ORIGINAL question and all retrieved candidate
    chunks once.

    coverage.py has already decided whether evidence actually
    SUPPORTS a requirement.

    These embeddings are now used only for:

        global question relevance
        redundancy calculation
        MMR filling

    They are NOT used to determine requirement support.
    """

    if not results:
        return []

    question_embedding = (
        embeddings.embed_query(
            question
        )
    )

    texts = [
        document.page_content
        for document in results
    ]

    document_embeddings = (
        embeddings.embed_documents(
            texts
        )
    )

    candidates = []

    for index, (
        document,
        document_embedding,
    ) in enumerate(
        zip(
            results,
            document_embeddings,
        )
    ):

        global_relevance = (
            cosine_similarity(
                question_embedding,
                document_embedding,
            )
        )

        candidates.append(
            {
                "index": index,
                "document": document,
                "embedding": (
                    document_embedding
                ),
                "global_relevance": (
                    global_relevance
                ),
                "characters": len(
                    document.page_content
                ),
                "document_key": (
                    get_document_key(
                        document
                    )
                ),
            }
        )

    return candidates


# ============================================================
# REDUNDANCY
# ============================================================

def calculate_redundancy(
    candidate_embedding,
    selected_embeddings,
):
    """
    Calculate maximum similarity between a candidate and any
    chunk already selected.

    High value = redundant.
    Low value = adds different information.
    """

    if not selected_embeddings:
        return 0.0

    similarities = [
        cosine_similarity(
            candidate_embedding,
            selected_embedding,
        )
        for selected_embedding
        in selected_embeddings
    ]

    return max(similarities)


# ============================================================
# MMR
# ============================================================

def calculate_mmr_score(
    relevance_score,
    redundancy_score,
):
    """
    Maximum Marginal Relevance.

    MMR =
        lambda * relevance
        -
        (1 - lambda) * redundancy
    """

    return (
        GENERATION_MMR_LAMBDA
        * relevance_score
        -
        (
            1
            - GENERATION_MMR_LAMBDA
        )
        * redundancy_score
    )


# ============================================================
# BUDGET CHECK
# ============================================================

def candidate_fits_budget(
    candidate,
    selected_count,
    selected_characters,
):
    """
    Check deterministic generation-context limits.
    """

    if (
        selected_count
        >= MAX_GENERATION_CHUNKS
    ):
        return False

    if (
        selected_characters
        + candidate["characters"]
        > MAX_GENERATION_CONTEXT_CHARS
    ):
        return False

    return True


# ============================================================
# LOOKUP CANDIDATE BY DOCUMENT
# ============================================================

def find_candidate_for_document(
    candidates,
    document,
):
    """
    Resolve a Document returned by coverage.py back to the
    prepared candidate containing its embedding and scores.

    We match using our deterministic document key.
    """

    target_key = (
        get_document_key(
            document
        )
    )

    for candidate in candidates:

        if (
            candidate["document_key"]
            == target_key
        ):
            return candidate

    return None


# ============================================================
# PHASE 1:
# ATTRIBUTION-AWARE COVERAGE SELECTION
# ============================================================

def select_attributed_evidence(
    requirements,
    attribution_map,
    candidates,
):
    """
    Select evidence that coverage.py has judged to actually
    SUPPORT each requirement.

    This replaces the old:

        requirement
            ↓
        cosine similarity
            ↓
        highest-scoring chunk

    architecture.

    New architecture:

        requirement
            ↓
        coverage.py support judgment
            ↓
        supporting candidates
            ↓
        choose best candidate among SUPPORTING evidence

    Embeddings may rank supporting passages, but they are no
    longer allowed to decide whether a passage is evidence.
    """

    selected_candidates = []

    selected_indexes = set()

    selected_characters = 0

    print(
        "\n"
        "------------------------------------------"
    )

    print(
        "GENERATION EVIDENCE - "
        "ATTRIBUTED COVERAGE SELECTION"
    )

    print(
        "------------------------------------------"
    )

    for requirement in requirements:

        requirement_id = (
            requirement["id"]
        )

        requirement_text = (
            requirement[
                "requirement"
            ]
        )

        attribution = (
            attribution_map.get(
                requirement_id,
                {},
            )
        )

        supporting_documents = (
            attribution.get(
                "documents",
                [],
            )
        )

        print()
        print(
            f"{requirement_id}: "
            f"{requirement_text}"
        )

        # ----------------------------------------------------
        # coverage.py found no directly supporting evidence.
        # ----------------------------------------------------

        if not supporting_documents:

            print(
                "Attributed evidence: NONE ❌"
            )

            continue

        # ----------------------------------------------------
        # Resolve attributed Documents back into candidates.
        # ----------------------------------------------------

        supporting_candidates = []

        for document in (
            supporting_documents
        ):

            candidate = (
                find_candidate_for_document(
                    candidates,
                    document,
                )
            )

            if candidate is None:
                continue

            supporting_candidates.append(
                candidate
            )

        if not supporting_candidates:

            print(
                "Attributed documents could not "
                "be resolved to candidates. ❌"
            )

            continue

        # ----------------------------------------------------
        # Rank ONLY the genuinely supporting candidates by
        # relevance to the ORIGINAL question.
        #
        # Important distinction:
        #
        # LLM:
        #     Does this support the requirement?
        #
        # Embeddings:
        #     Among supporting evidence, which is most
        #     relevant to the overall question?
        # ----------------------------------------------------

        supporting_candidates.sort(
            key=lambda candidate: (
                candidate[
                    "global_relevance"
                ]
            ),
            reverse=True,
        )

        chosen_candidate = None

        # ----------------------------------------------------
        # Prefer an unselected supporting chunk.
        #
        # If one chunk supports several requirements, we do
        # not need to add it repeatedly.
        # ----------------------------------------------------

        for candidate in (
            supporting_candidates
        ):

            if (
                candidate["index"]
                in selected_indexes
            ):
                continue

            if not candidate_fits_budget(
                candidate,
                len(selected_candidates),
                selected_characters,
            ):
                continue

            chosen_candidate = (
                candidate
            )

            break

        # ----------------------------------------------------
        # All supporting evidence may already be represented.
        #
        # This is NOT a failure.
        #
        # Example:
        #
        # p58 may support both R2 and R3.
        # If R2 already selected it, R3 still has supporting
        # evidence in the context.
        # ----------------------------------------------------

        if chosen_candidate is None:

            already_selected_support = [
                candidate
                for candidate
                in supporting_candidates
                if (
                    candidate["index"]
                    in selected_indexes
                )
            ]

            if already_selected_support:

                pages = [
                    str(
                        get_page(
                            candidate[
                                "document"
                            ]
                        )
                    )
                    for candidate
                    in already_selected_support
                ]

                print(
                    "Requirement already represented "
                    "by selected evidence."
                )

                print(
                    "Selected supporting page(s): "
                    + ", ".join(pages)
                )

                print(
                    "Decision: ALREADY COVERED ✅"
                )

            else:

                print(
                    "Supporting evidence exists, "
                    "but none fits remaining budget."
                )

                print(
                    "Decision: NOT ADDED ⚠️"
                )

            continue

        # ----------------------------------------------------
        # Add the chosen supporting candidate.
        # ----------------------------------------------------

        selected_candidates.append(
            chosen_candidate
        )

        selected_indexes.add(
            chosen_candidate["index"]
        )

        selected_characters += (
            chosen_candidate[
                "characters"
            ]
        )

        document = (
            chosen_candidate[
                "document"
            ]
        )

        print(
            f"Selected page: "
            f"{get_page(document)}"
        )

        print(
            f"Global question similarity: "
            f"{chosen_candidate['global_relevance']:.4f}"
        )

        print(
            f"Characters: "
            f"{chosen_candidate['characters']}"
        )

        print(
            f"Preview: "
            f"{get_preview(document)}"
        )

        print(
            "Decision: "
            "KEEP ATTRIBUTED EVIDENCE ✅"
        )

    return (
        selected_candidates,
        selected_indexes,
        selected_characters,
    )


# ============================================================
# PHASE 2:
# MMR FILL
# ============================================================

def fill_remaining_with_mmr(
    candidates,
    selected_candidates,
    selected_indexes,
    selected_characters,
):
    """
    After requirement-supporting evidence has been selected,
    use MMR to fill any remaining context capacity.

    This preserves the useful part of our previous MMR
    experiment.

    Order of authority is now:

        support
            >
        relevance/diversity
    """

    print(
        "\n"
        "------------------------------------------"
    )

    print(
        "GENERATION EVIDENCE - "
        "MMR FILL"
    )

    print(
        "------------------------------------------"
    )

    selected_embeddings = [
        candidate["embedding"]
        for candidate
        in selected_candidates
    ]

    remaining_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate["index"]
            not in selected_indexes
        )
    ]

    selection_round = 1

    while (
        len(selected_candidates)
        < MAX_GENERATION_CHUNKS
    ):

        best_candidate = None

        best_redundancy = None

        best_mmr_score = None

        for candidate in (
            remaining_candidates
        ):

            if not candidate_fits_budget(
                candidate,
                len(selected_candidates),
                selected_characters,
            ):
                continue

            redundancy_score = (
                calculate_redundancy(
                    candidate["embedding"],
                    selected_embeddings,
                )
            )

            mmr_score = (
                calculate_mmr_score(
                    candidate[
                        "global_relevance"
                    ],
                    redundancy_score,
                )
            )

            if (
                best_candidate is None
                or mmr_score
                > best_mmr_score
            ):

                best_candidate = (
                    candidate
                )

                best_redundancy = (
                    redundancy_score
                )

                best_mmr_score = (
                    mmr_score
                )

        if best_candidate is None:
            break

        selected_candidates.append(
            best_candidate
        )

        selected_indexes.add(
            best_candidate["index"]
        )

        selected_embeddings.append(
            best_candidate["embedding"]
        )

        selected_characters += (
            best_candidate[
                "characters"
            ]
        )

        remaining_candidates = [
            candidate
            for candidate
            in remaining_candidates
            if (
                candidate["index"]
                != best_candidate["index"]
            )
        ]

        document = (
            best_candidate[
                "document"
            ]
        )

        print()
        print(
            f"MMR fill round "
            f"{selection_round}"
        )

        print(
            f"Page: "
            f"{get_page(document)}"
        )

        print(
            f"Global relevance: "
            f"{best_candidate['global_relevance']:.4f}"
        )

        print(
            f"Redundancy: "
            f"{best_redundancy:.4f}"
        )

        print(
            f"MMR score: "
            f"{best_mmr_score:.4f}"
        )

        print(
            f"Characters: "
            f"{best_candidate['characters']}"
        )

        print(
            f"Preview: "
            f"{get_preview(document)}"
        )

        print(
            "Decision: KEEP AS FILL ✅"
        )

        selection_round += 1

    return (
        selected_candidates,
        selected_characters,
    )


# ============================================================
# DEBUG:
# PURE RELEVANCE RANKING
# ============================================================

def print_pure_relevance_ranking(
    candidates,
):
    """
    Keep displaying global cosine ranking.

    It is useful diagnostically because we can see cases such
    as p145 ranking well semantically while attribution rejects
    it as evidence.
    """

    print(
        "\n"
        "------------------------------------------"
    )

    print(
        "GENERATION EVIDENCE - "
        "PURE RELEVANCE RANKING"
    )

    print(
        "------------------------------------------"
    )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate[
                "global_relevance"
            ]
        ),
        reverse=True,
    )

    for rank, candidate in enumerate(
        ranked_candidates,
        start=1,
    ):

        document = (
            candidate["document"]
        )

        print()
        print(
            f"Candidate #{rank}"
        )

        print(
            f"Question similarity: "
            f"{candidate['global_relevance']:.4f}"
        )

        print(
            f"Page: "
            f"{get_page(document)}"
        )

        print(
            f"Preview: "
            f"{get_preview(document)}"
        )


# ============================================================
# MAIN EVIDENCE SELECTOR
# ============================================================

def select_generation_evidence(
    embeddings,
    question,
    requirements,
    attribution_map,
    results,
):
    """
    Select the evidence actually shown to the answer generator.

    FINAL ARCHITECTURE FOR THIS EXPERIMENT:

        retrieved candidate pool
                  ↓
        coverage.py attribution
                  ↓
        requirement -> supporting chunks
                  ↓
        PHASE 1
        guarantee supporting evidence representation
                  ↓
        PHASE 2
        MMR fills remaining context capacity
                  ↓
        max chunk / character budget
                  ↓
        generation context

    Crucially:

        embeddings DO NOT decide support.

    They are used only AFTER support attribution.
    """

    if not results:
        return []

    # --------------------------------------------------------
    # Prepare all candidates once.
    # --------------------------------------------------------

    candidates = (
        prepare_candidates(
            embeddings,
            question,
            results,
        )
    )

    # --------------------------------------------------------
    # Diagnostic baseline.
    # --------------------------------------------------------

    print_pure_relevance_ranking(
        candidates
    )

    # --------------------------------------------------------
    # PHASE 1:
    # Requirement-supporting evidence.
    # --------------------------------------------------------

    (
        selected_candidates,
        selected_indexes,
        selected_characters,
    ) = select_attributed_evidence(
        requirements,
        attribution_map,
        candidates,
    )

    attributed_selection_count = (
        len(selected_candidates)
    )

    # --------------------------------------------------------
    # PHASE 2:
    # Fill spare capacity with MMR.
    # --------------------------------------------------------

    (
        selected_candidates,
        selected_characters,
    ) = fill_remaining_with_mmr(
        candidates,
        selected_candidates,
        selected_indexes,
        selected_characters,
    )

    # --------------------------------------------------------
    # Return normal LangChain Documents expected by context.py.
    # --------------------------------------------------------

    selected_documents = [
        candidate["document"]
        for candidate
        in selected_candidates
    ]

    # --------------------------------------------------------
    # Final diagnostics.
    # --------------------------------------------------------

    print(
        "\n"
        "------------------------------------------"
    )

    print(
        "GENERATION CONTEXT SUMMARY"
    )

    print(
        "------------------------------------------"
    )

    print(
        f"Retrieved candidates: "
        f"{len(results)}"
    )

    print(
        f"Evidence requirements: "
        f"{len(requirements)}"
    )

    print(
        f"Attributed coverage chunks: "
        f"{attributed_selection_count}"
    )

    print(
        f"Final selected chunks: "
        f"{len(selected_documents)}"
    )

    print(
        f"Selected characters: "
        f"{selected_characters}"
    )

    print(
        f"Character budget: "
        f"{MAX_GENERATION_CONTEXT_CHARS}"
    )

    print(
        f"Chunk budget: "
        f"{MAX_GENERATION_CHUNKS}"
    )

    print(
        f"Generation MMR lambda: "
        f"{GENERATION_MMR_LAMBDA}"
    )

    return selected_documents