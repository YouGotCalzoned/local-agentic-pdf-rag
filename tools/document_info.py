from langchain_core.tools import tool

from agent_runtime import agent_runtime


@tool
def document_info():
    """
    Return metadata about the currently loaded document.

    Use this tool when the user asks about the loaded document itself,
    such as its filename, number of pages, number of non-empty pages,
    or number of indexed chunks.

    Do not use this tool to answer questions about the document's
    actual content. Use search_document for that.
    """

    if not agent_runtime.has_document():
        return {
            "success": False,
            "error": "No document is currently loaded.",
        }

    if agent_runtime.document_info is None:
        return {
            "success": False,
            "error": "Document metadata is unavailable.",
        }

    return {
        "success": True,
        "document": agent_runtime.document_info,
    }