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
        results = agent_runtime.vector_store.similarity_search_with_score(
            query,
            k=4,
        )

        passages = []

        for index, (document, score) in enumerate(results, start=1):

            page = document.metadata.get("page")

            # PyPDFLoader page metadata is zero-based.
            display_page = page + 1 if isinstance(page, int) else None

            passages.append(
                {
                    "passage_id": f"P{index}",
                    "page": display_page,
                    "distance": float(score),
                    "text": document.page_content,
                }
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