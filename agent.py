from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from tools.calculator import calculator
from tools.document_search import search_document
from tools.document_info import document_info

# ============================================================
# Configuration
# ============================================================

MAX_AGENT_STEPS = 5

AGENT_SYSTEM_PROMPT = """
You are a general-purpose assistant with access to tools.

Use tools when they are useful for answering the user's question.
You do not need to use a tool when you can answer directly.

Important rules for tool use:

- A tool call cannot directly use the result of another tool call
  made in the same step.
- If one tool depends on the result of another tool:
    1. Call the first tool.
    2. Wait for its result.
    3. Use that result to construct the next tool call.
- Never place a tool call or function syntax inside another tool's arguments.
- Use the calculator whenever arithmetic is required.
- Only use tools that are actually available.
"""

# ============================================================
# Tools
# ============================================================

tools = [
    calculator,
    search_document,
    document_info,
]

# Build a registry:
#
# {
#     "calculator": calculator
# }
#
# Later this can become:
#
# {
#     "calculator": calculator,
#     "search_document": search_document,
#     "document_info": document_info,
# }
#
tool_registry = {tool.name: tool for tool in tools}


# ============================================================
# Model
# ============================================================

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# Agent loop
# ============================================================


def run_agent(question: str):

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    # Collect a structured execution trace for the UI.
    trace = []

    for step in range(MAX_AGENT_STEPS):

        print(f"\n{'=' * 60}")
        print(f"AGENT STEP {step + 1}")
        print(f"{'=' * 60}")

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        print("\nLLM CONTENT:")
        print(response.content)

        print("\nTOOL CALLS:")
        print(response.tool_calls)

        # ----------------------------------------------------
        # No tool calls -> agent has finished.
        # ----------------------------------------------------

        if not response.tool_calls:

            trace.append(
                {
                    "step": step + 1,
                    "type": "final_answer",
                    "content": response.content,
                }
            )

            print("\nFINAL ANSWER:")
            print(response.content)

            return {
                "answer": response.content,
                "trace": trace,
                "completed": True,
            }

        # ----------------------------------------------------
        # Execute requested tools.
        # ----------------------------------------------------

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print(f"\nREQUESTED TOOL: {tool_name}")
            print(f"ARGUMENTS: {tool_args}")

            selected_tool = tool_registry.get(tool_name)

            if selected_tool is None:

                tool_result = {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                }

            else:

                try:
                    tool_result = selected_tool.invoke(tool_args)

                except Exception as exc:

                    tool_result = {
                        "success": False,
                        "error": str(exc),
                    }

            print("\nTOOL RESULT:")
            print(tool_result)

            # Record exactly what happened.
            trace.append(
                {
                    "step": step + 1,
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                }
            )

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )

    # --------------------------------------------------------
    # Agent exceeded its allowed number of reasoning steps.
    # --------------------------------------------------------

    failure_message = (
        "I could not complete the request within "
        "the allowed number of tool-use steps."
    )

    trace.append(
        {
            "step": MAX_AGENT_STEPS,
            "type": "max_steps",
            "content": failure_message,
        }
    )

    print("\nAgent reached maximum number of steps.")

    return {
        "answer": failure_message,
        "trace": trace,
        "completed": False,
    }


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":

    from agent_runtime import agent_runtime
    from document_service import process_pdf
    from models import load_embeddings

    PDF_PATH = "data/book.pdf"

    print("\nLoading document for agent...")

    embeddings = load_embeddings()

    vector_store, document_info = process_pdf(
        PDF_PATH,
        embeddings,
    )

    agent_runtime.set_document(
        vector_store=vector_store,
        document_info=document_info,
    )

    print(f"Loaded: {document_info['name']} " f"({document_info['chunks']} chunks)")

    question = input("\nAsk the agent something: ")

    run_agent(question)
