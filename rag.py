from config import (
    MAX_RETRIEVAL_ROUNDS,
)

from models import (
    load_embeddings,
    load_vector_store,
    load_llm,
)

from context import (
    build_context,
    build_labeled_context,
    build_source_map,
)

from retrieval import (
    initial_search,
    run_follow_up_searches,
)

from planner import (
    derive_evidence_requirements,
    evaluate_requirement_coverage,
    all_requirements_covered,
    plan_follow_up_searches,
)

from evidence import (
    select_generation_evidence,
)

from generation import (
    generate_draft_answer,
    generate_final_answer,
)

from verification import (
    extract_claims,
    verify_claims,
)

from coverage import (
    attribute_evidence_to_requirements,
    build_attribution_coverage_state,
    print_attribution_summary,
)

# ============================================================
# DEBUG PRINTING
# ============================================================


def print_requirements(
    requirements,
):
    """
    Show the evidence plan derived from the original question.
    """

    print("\n" "==========================================")

    print("EVIDENCE REQUIREMENTS")

    print("==========================================")

    for item in requirements:

        print(f"{item['id']}: " f"{item['requirement']}")


def print_coverage_state(
    coverage_state,
):
    """
    Show current evidence coverage.

    Example:

        R1 COVERED
        R2 PARTIAL
        R3 MISSING
    """

    print("\n" "------------------------------------------")

    print("EVIDENCE COVERAGE")

    print("------------------------------------------")

    for item in coverage_state:

        print(f"{item['id']} | " f"{item['status']}")

        print(f"Requirement: " f"{item['requirement']}")

        if item["reason"]:

            print(f"Reason: " f"{item['reason']}")

        print()


# ============================================================
# AGENTIC RETRIEVAL
# ============================================================


def run_agentic_retrieval(
    llm,
    vector_store,
    question,
):
    """
    Perform bounded, coverage-aware agentic retrieval.

    New architecture:

        original question
              ↓
        derive evidence requirements
              ↓
        initial retrieval
              ↓
        relevance threshold
              ↓
        initial retrieval MMR
              ↓
        evaluate requirement coverage
              ↓
        all covered?
          /       \\
        yes       no
        |          |
       stop    search missing
                  |
                  ↓
            retrieve evidence
                  |
                  ↓
            reassess coverage

    Python owns:

        requirement IDs
        retrieval rounds
        accumulated evidence
        search history
        stop condition

    LLM owns:

        requirement semantics
        coverage judgment
        search planning
    """

    # --------------------------------------------------------
    # Stage 0:
    # Derive the evidence plan BEFORE retrieval begins.
    # --------------------------------------------------------

    requirements = derive_evidence_requirements(
        llm,
        question,
    )

    print_requirements(requirements)

    # --------------------------------------------------------
    # Stage 1:
    # Initial retrieval.
    #
    # initial_search() already performs:
    #
    # similarity search
    #       ↓
    # distance threshold
    #       ↓
    # retrieval-stage MMR
    # --------------------------------------------------------

    print("\nInitial retrieval...")

    all_results = initial_search(
        vector_store,
        question,
    )

    # --------------------------------------------------------
    # Deterministic out-of-domain exit.
    # --------------------------------------------------------

    if not all_results:

        print("\n" "==========================================")

        print("OUT-OF-DOMAIN / " "INSUFFICIENT EVIDENCE")

        print("==========================================")

        print(
            "\nNo sufficiently relevant "
            "information was found in "
            "the knowledge base."
        )

        return (
            [],
            requirements,
            [],
        )

    # --------------------------------------------------------
    # Agent state.
    # --------------------------------------------------------

    previous_searches = []

    coverage_state = []

    # --------------------------------------------------------
    # Coverage-aware retrieval loop.
    # --------------------------------------------------------

    for round_number in range(
        1,
        MAX_RETRIEVAL_ROUNDS + 1,
    ):

        print("\n" "==========================================")

        print(f"RETRIEVAL ROUND " f"{round_number}")

        print("==========================================")

        # ----------------------------------------------------
        # Build the complete CURRENT retrieval context.
        #
        # This context is for planning only.
        #
        # It is NOT necessarily the context that the final
        # answer generator will receive.
        # ----------------------------------------------------

        planner_context = build_context(all_results)

        # ----------------------------------------------------
        # Evaluate each evidence requirement separately.
        # ----------------------------------------------------

        coverage_state = evaluate_requirement_coverage(
            llm,
            question,
            requirements,
            planner_context,
        )

        print_coverage_state(coverage_state)

        # ----------------------------------------------------
        # Deterministic stopping rule.
        #
        # The LLM does NOT decide "stop".
        #
        # Python checks:
        #
        #     every requirement == COVERED
        # ----------------------------------------------------

        if all_requirements_covered(coverage_state):

            print("\nAll evidence requirements " "are covered.")

            print("Stopping retrieval.")

            break

        # ----------------------------------------------------
        # Ask planner to search ONLY for PARTIAL / MISSING
        # requirements.
        # ----------------------------------------------------

        search_queries = plan_follow_up_searches(
            llm,
            question,
            coverage_state,
            planner_context,
            previous_searches,
        )

        # ----------------------------------------------------
        # Planner could not generate anything useful.
        # ----------------------------------------------------

        if not search_queries:

            print("\nNo new useful follow-up " "searches were generated.")

            print("Stopping retrieval.")

            break

        print("\nFOLLOW-UP SEARCHES:")

        for search_query in search_queries:

            print(f"- {search_query}")

        # ----------------------------------------------------
        # Save search history BEFORE next planning round.
        # ----------------------------------------------------

        previous_searches.extend(search_queries)

        # ----------------------------------------------------
        # Execute vector searches.
        # ----------------------------------------------------

        (
            updated_results,
            added_count,
        ) = run_follow_up_searches(
            vector_store,
            all_results,
            search_queries,
        )

        all_results = updated_results

        print(f"\nAdded " f"{added_count} " f"new unique chunks.")

        print(f"Total unique chunks: " f"{len(all_results)}")

        # ----------------------------------------------------
        # If searches retrieved nothing new, another round
        # cannot improve coverage using the same strategy.
        # ----------------------------------------------------

        if added_count == 0:

            print("\nNo new evidence " "was found.")

            print("Stopping retrieval.")

            break

    # --------------------------------------------------------
    # FINAL COVERAGE CHECK
    #
    # This is important.
    #
    # Imagine round 3 retrieves the missing evidence.
    #
    # The loop would otherwise end because MAX_RETRIEVAL_ROUNDS
    # has been reached WITHOUT evaluating those newly retrieved
    # chunks.
    #
    # So we perform one final coverage assessment.
    # --------------------------------------------------------

    final_planner_context = build_context(all_results)

    coverage_state = evaluate_requirement_coverage(
        llm,
        question,
        requirements,
        final_planner_context,
    )

    print("\n" "==========================================")

    print("FINAL RETRIEVAL COVERAGE")

    print("==========================================")

    print_coverage_state(coverage_state)

    return (
        all_results,
        requirements,
        coverage_state,
    )


# ============================================================
# FULL QUESTION-ANSWERING PIPELINE
# ============================================================


def answer_question(
    llm,
    vector_store,
    embeddings,
    question,
):
    """
    Full RAG pipeline.

    Current architecture:

        question
            ↓
        evidence requirements
            ↓
        initial retrieval
            ↓
        threshold gate
            ↓
        initial MMR
            ↓
        coverage-aware agentic retrieval
            ↓
        accumulated candidate evidence
            ↓
        generation-stage evidence selection
            ↓
        context budget
            ↓
        labeled context [S1...Sn]
            ↓
        grounded draft
            ↓
        atomic claim extraction
            ↓
        citation-scoped verification
            ↓
        deterministic filtering
            ↓
        final answer
    """

    # --------------------------------------------------------
    # Stage 1:
    # Coverage-aware agentic retrieval.
    # --------------------------------------------------------

    (
        results,
        requirements,
        coverage_state,
    ) = run_agentic_retrieval(
        llm,
        vector_store,
        question,
    )

    # --------------------------------------------------------
    # Deterministic out-of-domain exit.
    # --------------------------------------------------------

    if not results:

        print("\n" "==========================================")

        print("FINAL ANSWER")

        print("==========================================")

        print(
            "\nThe knowledge base does not "
            "contain sufficiently relevant "
            "information to answer this "
            "question."
        )

        return {
            "answer": (
                "The knowledge base does not contain sufficiently "
                "relevant information to answer this question."
            ),
            "sources": [],
            "requirements": requirements,
            "coverage_state": coverage_state,
            "verified_claims": [],
        }

    # --------------------------------------------------------
    # Evidence attribution.
    #
    # Planner told us WHAT evidence is required.
    # Retrieval found candidate evidence.
    #
    # coverage.py now determines WHICH retrieved chunks
    # actually support each requirement.
    # --------------------------------------------------------

    attribution_map = attribute_evidence_to_requirements(
        llm,
        question,
        requirements,
        results,
    )

    print_attribution_summary(
        requirements,
        attribution_map,
    )
    # --------------------------------------------------------
    # Stage 2:
    # Generation evidence selection.
    #
    # IMPORTANT:
    #
    # We are NOT making this coverage-aware yet.
    #
    # This remains the MMR evidence.py implementation from
    # our previous experiment.
    #
    # That is deliberate.
    # --------------------------------------------------------

    generation_results = select_generation_evidence(
        embeddings,
        question,
        requirements,
        attribution_map,
        results,
    )

    # --------------------------------------------------------
    # Defensive check.
    # --------------------------------------------------------

    if not generation_results:

        print("\n" "==========================================")

        print("FINAL ANSWER")

        print("==========================================")

        print(
            "\nRelevant evidence was retrieved, "
            "but no evidence survived the "
            "generation-context selection stage."
        )

        return {
            "answer": (
                "Relevant evidence was retrieved, but no evidence "
                "survived the generation-context selection stage."
            ),
            "sources": [],
            "requirements": requirements,
            "coverage_state": coverage_state,
            "verified_claims": [],
        }

    # --------------------------------------------------------
    # Stage 3:
    # Create fresh S1...Sn labels ONLY from generation
    # evidence.
    #
    # This keeps:
    #
    # labeled context
    # verifier
    # source map
    #
    # in the SAME source namespace.
    # --------------------------------------------------------

    labeled_context = build_labeled_context(generation_results)

    # --------------------------------------------------------
    # Stage 4:
    # Generate grounded draft.
    # --------------------------------------------------------

    draft_answer = generate_draft_answer(
        llm,
        question,
        labeled_context,
    )

    print("\n" "==========================================")

    print("DRAFT ANSWER")

    print("==========================================")

    print(f"\n{draft_answer}")

    # --------------------------------------------------------
    # Stage 5:
    # Extract atomic claims.
    # --------------------------------------------------------

    claims = extract_claims(
        llm,
        draft_answer,
    )

    print("\n" "==========================================")

    print("EXTRACTED CLAIMS")

    print("==========================================")

    print()

    if not claims:

        print("No factual claims " "were extracted.")

    else:

        for index, claim in enumerate(
            claims,
            start=1,
        ):

            print(f"{index}. {claim}")

    # --------------------------------------------------------
    # Stage 6:
    # Citation-scoped verification.
    #
    # verification.py:
    #
    # claim [S2]
    #      ↓
    # Python resolves S2
    #      ↓
    # verifier receives ONLY S2 evidence
    #      ↓
    # SUPPORTED / UNSUPPORTED
    # --------------------------------------------------------

    (
        verified_claims,
        verification_debug,
    ) = verify_claims(
        llm,
        claims,
        labeled_context,
    )

    print("\n" "==========================================")

    print("CLAIM VERIFICATION")

    print("==========================================")

    print()

    for item in verification_debug:

        index = item["index"]

        claim = item["claim"]

        verification_text = item["verification_text"]

        kept = item["kept"]

        print(f"Claim {index}: " f"{claim}")

        print(f"Verifier: " f"{verification_text}")

        if kept:

            print("Decision: KEEP ✅")

        else:

            print("Decision: DISCARD ❌")

        print()

    # --------------------------------------------------------
    # Stage 7:
    # Show verified facts.
    # --------------------------------------------------------

    print("\n" "==========================================")

    print("VERIFIED FACTS")

    print("==========================================")

    print()

    if not verified_claims:

        print("No claims survived " "grounding verification.")

    else:

        for item in verified_claims:

            claim = item["claim"]

            sources = item["sources"]

            print(f"- {claim} " f"[{sources}]")

    # --------------------------------------------------------
    # Stage 8:
    # Final answer synthesis.
    #
    # The final generator receives ONLY claims that survived
    # citation-scoped verification.
    # --------------------------------------------------------

    final_answer = generate_final_answer(
        llm,
        question,
        verified_claims,
    )

    print("\n" "==========================================")

    print("VERIFIED FINAL ANSWER")

    print("==========================================")

    print(f"\n{final_answer}")

    # --------------------------------------------------------
    # Stage 9:
    # Show final coverage state.
    #
    # This gives us useful diagnostics when the final answer
    # is incomplete.
    # --------------------------------------------------------

    print("\n" "==========================================")

    print("FINAL EVIDENCE REQUIREMENT STATE")

    print("==========================================")

    for item in coverage_state:

        print(f"{item['id']} | " f"{item['status']} | " f"{item['requirement']}")

    # --------------------------------------------------------
    # Stage 10:
    # Source map.
    #
    # CRITICAL:
    #
    # Source map MUST use generation_results,
    # NOT every chunk discovered during retrieval.
    #
    # Otherwise [S#] labels stop matching the generator /
    # verifier evidence namespace.
    # --------------------------------------------------------

    source_map = build_source_map(generation_results)

    print("\n" "------------------------------------------")

    print("SOURCE MAP")

    print("------------------------------------------")

    for item in source_map:

        print(item)


        # --------------------------------------------------------
    # Build structured source information for UI clients.
    #
    # Internal page metadata is zero-based.
    # The UI should display human-readable one-based pages.
    # --------------------------------------------------------

    ui_sources = []

    for index, document in enumerate(
        generation_results,
        start=1,
    ):

        internal_page = (
            document.metadata.get(
                "page"
            )
        )

        if isinstance(
            internal_page,
            int,
        ):

            display_page = (
                internal_page + 1
            )

        else:

            display_page = (
                internal_page
            )

        display_name = (
            document.metadata.get(
                "display_name"
            )
            or document.metadata.get(
                "source"
            )
            or "Uploaded PDF"
        )

        ui_sources.append(
            {
                "id": f"S{index}",
                "document": display_name,
                "page": display_page,
            }
        )
    # --------------------------------------------------------
    # Return structured result for callers such as Streamlit.
    #
    # CLI mode can simply ignore this return value.
    # --------------------------------------------------------

    return {
        "answer": final_answer,
        "sources": ui_sources,
        "requirements": requirements,
        "coverage_state": coverage_state,
        "verified_claims": verified_claims,
    }


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================


def main():
    """
    Interactive local RAG application.
    """

    # --------------------------------------------------------
    # Load these ONCE.
    #
    # Do not reload embeddings / FAISS / Ollama for every
    # question.
    # --------------------------------------------------------

    embeddings = load_embeddings()

    vector_store = load_vector_store(embeddings)

    llm = load_llm()

    print("\n" "==========================================")

    print("Coverage-Aware + " "Claim-Verified RAG ready")

    print("==========================================")

    print("Type 'exit' to quit.\n")

    # --------------------------------------------------------
    # Interactive loop.
    # --------------------------------------------------------

    while True:

        question = input("Ask a question: ").strip()

        if not question:
            continue

        if question.lower() == "exit":

            print("Goodbye.")

            break

        try:

            answer_question(
                llm,
                vector_store,
                embeddings,
                question,
            )

        except Exception as exception:

            print("\nAn error occurred " "while processing " "the question:")

            print(exception)

        print()


if __name__ == "__main__":
    main()
