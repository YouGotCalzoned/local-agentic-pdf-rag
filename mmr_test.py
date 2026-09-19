from models import load_embeddings, load_vector_store


def print_documents(title, documents):
    print("\n==========================================")
    print(title)
    print("==========================================\n")

    for index, document in enumerate(documents, start=1):
        page = document.metadata.get("page")

        preview = (
            document.page_content
            .replace("\n", " ")
            .strip()
        )

        preview = preview[:400]

        print(f"Result #{index}")
        print(f"Page: {page}")
        print(f"Preview: {preview}")
        print()


def main():
    embeddings = load_embeddings()

    vector_store = load_vector_store(
        embeddings
    )

    query = input(
        "Enter a test query: "
    ).strip()

    if not query:
        print("No query entered.")
        return

    # ---------------------------------------------------------
    # Normal similarity search
    # ---------------------------------------------------------

    normal_results = (
        vector_store.similarity_search(
            query,
            k=3,
        )
    )

    print_documents(
        "NORMAL SIMILARITY SEARCH",
        normal_results,
    )

    # ---------------------------------------------------------
    # Maximum Marginal Relevance search
    # ---------------------------------------------------------
    #
    # fetch_k = number of candidates initially considered
    # k       = number of final documents returned
    #
    # lambda_mult controls relevance vs diversity:
    #
    # 1.0 -> mostly relevance
    # 0.5 -> balanced
    # 0.0 -> mostly diversity
    #
    # We start with 0.5 so we can clearly observe the effect.
    # ---------------------------------------------------------

    mmr_results = (
        vector_store.max_marginal_relevance_search(
            query,
            k=3,
            fetch_k=15,
            lambda_mult=0.2,
        )
    )

    print_documents(
        "MMR SEARCH",
        mmr_results,
    )


if __name__ == "__main__":
    main()