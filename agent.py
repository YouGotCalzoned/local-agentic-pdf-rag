from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from tools.calculator import calculator


# ============================================================
# Configuration
# ============================================================

MAX_AGENT_STEPS = 5


# ============================================================
# Tools
# ============================================================

tools = [
    calculator,
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
tool_registry = {
    tool.name: tool
    for tool in tools
}


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
        HumanMessage(content=question)
    ]

    for step in range(MAX_AGENT_STEPS):

        print(f"\n{'=' * 60}")
        print(f"AGENT STEP {step + 1}")
        print(f"{'=' * 60}")

        # ----------------------------------------------------
        # Ask the LLM what it wants to do next.
        # ----------------------------------------------------

        response = llm_with_tools.invoke(messages)

        messages.append(response)

        print("\nLLM CONTENT:")
        print(response.content)

        print("\nTOOL CALLS:")
        print(response.tool_calls)

        # ----------------------------------------------------
        # No tool calls means the model has finished.
        # ----------------------------------------------------

        if not response.tool_calls:

            print("\nFINAL ANSWER:")
            print(response.content)

            return response.content

        # ----------------------------------------------------
        # Execute every requested tool.
        # ----------------------------------------------------

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print(f"\nREQUESTED TOOL: {tool_name}")
            print(f"ARGUMENTS: {tool_args}")

            # ------------------------------------------------
            # Validate that the requested tool actually exists.
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Return the observation to the LLM.
            # ------------------------------------------------

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )

    # ========================================================
    # Safety stop
    # ========================================================

    print("\nAgent reached maximum number of steps.")

    return (
        "I could not complete the request within "
        "the allowed number of tool-use steps."
    )


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":

    question = input("\nAsk the agent something: ")

    run_agent(question)