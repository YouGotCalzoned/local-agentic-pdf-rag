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



def validate_final_answer(
    llm,
    question: str,
    tools_used: list[str],
) -> dict:
    """
    Validate whether required deterministic arithmetic was performed
    using the calculator tool.

    The LLM decides only whether the user's request requires arithmetic.
    Python deterministically checks whether calculator was actually used.
    """

    prompt = f"""
You are validating an AI agent request.

Original user question:
{question}

Determine whether answering the user's request requires arithmetic
or numerical calculation.

Return exactly one of:

ARITHMETIC_REQUIRED
NO_ARITHMETIC

Do not provide any explanation.
"""

    response = llm.invoke(prompt)

    decision = response.content.strip()

    # Semantic judgment: let the LLM decide whether arithmetic is required.
    arithmetic_required = decision == "ARITHMETIC_REQUIRED"

    # Deterministic fact: Python knows exactly which tools were used.
    calculator_used = "calculator" in tools_used

    # Retry only when arithmetic is required AND calculator was not used.
    retry_calculator = (
        arithmetic_required
        and not calculator_used
    )

    return {
        "decision": decision,
        "arithmetic_required": arithmetic_required,
        "calculator_used": calculator_used,
        "retry_calculator": retry_calculator,
    }

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
    tools_used = []
    calculator_retry_used = False

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

    # --------------------------------------------------------
    # Validate whether the agent skipped the calculator.
    # --------------------------------------------------------

            validation = validate_final_answer(
                llm=llm,
                question=question,
                tools_used=tools_used,
            )

            print("\nVALIDATION:")
            print(validation)
            trace.append(
                {
                    "step": step + 1,
                    "type": "validation_debug",
                    "decision": validation["decision"],
                    "arithmetic_required": validation["arithmetic_required"],
                    "calculator_used": validation["calculator_used"],
                    "retry_calculator": validation["retry_calculator"],
                    "tools_used": list(tools_used),
                }
            )
            # --------------------------------------------------------
            # If arithmetic was required but calculator was skipped,
            # reject this answer and give the agent one retry.
            # --------------------------------------------------------

            if (
                validation["retry_calculator"]
                and not calculator_retry_used
            ):

                calculator_retry_used = True

                trace.append(
                    {
                        "step": step + 1,
                        "type": "validation_retry",
                        "reason": (
                            "Arithmetic was required but the "
                            "calculator tool was not used."
                        ),
                    }
                )

                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous response cannot be accepted "
                            "because arithmetic was required but you did "
                            "not use the calculator tool. "
                            "Use the calculator tool for the required "
                            "calculation, then answer the original question."
                        )
                    )
                )

                # Continue around the agent loop.
                continue

            # --------------------------------------------------------
            # Validation passed, or the single retry was already used.
            # Accept the final response.
            # --------------------------------------------------------

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
            tools_used.append(tool_name)
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
