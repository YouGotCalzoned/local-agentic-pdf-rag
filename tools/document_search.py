from langchain_core.tools import tool

from agent_runtime import agent_runtime


@tool
def search_document(query: str):
    """
    Search the currently loaded PDF for information relevant to a query.

    Use this tool when the user's question requires information from
    the uploaded document.

    Args:
        query: A focused semantic search query describing the information
               needed from the document.
    """

    if not agent_runtime.has_document():
        return {
            "success": False,
            "error": "No document is currently loaded.",
        }

    try:
        results = (
            agent_runtime.vector_store.similarity_search_with_score(
                query,
                k=4,
            )
        )

        passages = []

        for document, score in results:

            page = document.metadata.get("page")

            display_page = (
                page + 1
                if isinstance(page, int)
                else None
            )

            # Use the chunk's existing metadata if available.
            chunk_id = document.metadata.get("chunk_id")

            # Fall back to a deterministic page/content-based ID
            # if the ingestion pipeline does not yet provide one.
            if chunk_id is not None:
                source_id = f"C{chunk_id}"
            else:
                source_id = (
                    f"PAGE_{display_page}"
                    if display_page is not None
                    else "UNKNOWN"
                )

            passage = {
                "source_id": source_id,
                "document": (
                    agent_runtime.document_info.get(
                        "name",
                        "Unknown document",
                    )
                ),
                "page": display_page,
                "distance": float(score),
                "text": document.page_content,
            }

    # Return the passage to the agent as before.
            passages.append(passage)

    # Also store it in the runtime evidence pool.
    # source_id allows the runtime to deduplicate chunks
    # retrieved by multiple searches.
            agent_runtime.add_evidence(
                source_id,
                passage,
            )

        return {
            "success": True,
            "query": query,
            "passages": passages,
        }

    except Exception as exc:
        return {
            "success": False,
            "query": query,
            "error": str(exc),
        }