from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from tools.calculator import calculator
from tools.document_search import search_document
from tools.document_info import document_info
from agent_runtime import agent_runtime
from planner import (
    derive_evidence_requirements,
    evaluate_requirement_coverage,
    evaluate_agent_requirement_coverage,
    all_requirements_covered,
    get_uncovered_requirements,
    plan_follow_up_searches,
)
# ============================================================
# Configuration
# ============================================================

MAX_AGENT_STEPS = 10

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


Document retrieval:

When answering questions about the uploaded document:

- Break multi-part questions into distinct information needs.
- Use search_document to gather evidence for those needs.
- After each search, inspect the returned passages and determine
  whether the original question is fully supported.
- If important parts of the question are still unsupported, call
  search_document again with a focused query for the missing evidence.
- Do not assume that one broad search is sufficient for a multi-part question.
- Prefer focused searches over combining many unrelated concepts into
  one search query.
- Answer only from evidence returned by document tools.
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


def build_agent_evidence_context(evidence):
    """
    Convert accumulated agent evidence into labeled text
    for semantic coverage evaluation.
    """

    sections = []

    for passage in evidence:

        source_id = passage["source_id"]
        page = passage.get("page")
        text = passage["text"]

        sections.append(
            f"[{source_id}] Page {page}\n{text}"
        )

    return "\n\n".join(sections)

def merge_monotonic_coverage(
    previous_state,
    new_state,
):
    """
    Merge a newly evaluated coverage state with the best
    coverage previously observed.

    Coverage may improve:

        MISSING -> PARTIAL -> COVERED

    but may never move backwards.
    """

    status_rank = {
        "MISSING": 0,
        "PARTIAL": 1,
        "COVERED": 2,
    }

    # Index the previous state by requirement ID.
    previous_by_id = {
        item["id"]: item
        for item in previous_state
    }

    merged = []

    for new_item in new_state:

        requirement_id = new_item["id"]

        previous_item = previous_by_id.get(
            requirement_id
        )

        # First evaluation of this requirement.
        if previous_item is None:
            merged.append(new_item)
            continue

        previous_status = previous_item.get(
            "status",
            "MISSING",
        ).upper()

        new_status = new_item.get(
            "status",
            "MISSING",
        ).upper()

        previous_rank = status_rank.get(
            previous_status,
            0,
        )

        new_rank = status_rank.get(
            new_status,
            0,
        )

        # Keep whichever assessment represents
        # stronger evidence coverage.
        if new_rank >= previous_rank:
            merged.append(new_item)
        else:
            merged.append(previous_item)

    return merged

def generate_best_effort_answer(
    question,
    evidence,
    coverage_state,
):
    """
    Generate the strongest grounded answer possible when the
    agent reaches its reasoning/retrieval budget before all
    evidence requirements are marked COVERED.
    """

    evidence_context = build_agent_evidence_context(
        evidence
    )

    coverage_text = "\n".join(
        (
            f"- {item['id']} [{item['status']}]: "
            f"{item['requirement']}"
        )
        for item in coverage_state
    )

    prompt = f"""
You are generating the final answer for a
Retrieval-Augmented Generation system.

The retrieval agent has reached its search budget.

Answer the ORIGINAL USER QUESTION using ONLY the accumulated
retrieved evidence below.

ORIGINAL USER QUESTION:

{question}


CURRENT EVIDENCE COVERAGE:

{coverage_text}


ACCUMULATED RETRIEVED EVIDENCE:

{evidence_context}


Rules:

- Give the strongest useful answer supported by the evidence.
- Use ONLY information contained in the accumulated evidence.
- Do not use outside or pretrained knowledge.
- Do not invent information to fill gaps.
- Some evidence requirements may still be PARTIAL or MISSING.
- PARTIAL does not mean the available evidence is useless.
  Use whatever that evidence supports.
- If an important part of the original question genuinely
  cannot be answered from the evidence, state that limitation
  briefly.
- Do not discuss the retrieval process, coverage evaluator,
  search budget, or internal requirement IDs.
- Do not mention that you are producing a "best-effort" answer.
- Answer the user naturally and directly.
"""

    response = llm.invoke(prompt)

    return response.content

# ============================================================
# Agent loop
# ============================================================


def run_agent(question: str):

    # Each agent execution gets its own evidence pool.
    # Evidence from a previous question must never leak
    # into the current question.
    agent_runtime.reset_evidence()

    # Track document searches attempted during this agent run.
    previous_searches = []

    # Track consecutive document searches that produce
    # no new unique evidence.
    consecutive_stagnant_searches = 0

    # Persist the strongest coverage state reached for each
    # requirement during this agent run.
    best_coverage_state = []

    # Build an explicit evidence plan for the original question.
    requirements = derive_evidence_requirements(
        llm,
        question,
    )

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
        # No tool calls -> agent has proposed a final answer.
        # ----------------------------------------------------

        if not response.tool_calls:

            # ------------------------------------------------
            # DOCUMENT EVIDENCE COVERAGE CHECK
            # ------------------------------------------------

            if "search_document" in tools_used:

                evidence = agent_runtime.get_evidence()

                evidence_context = (
                    build_agent_evidence_context(
                        evidence
                    )
                )

                valid_evidence_ids = {
                    passage["source_id"]
                    for passage in evidence
                }

                new_coverage_state = (
                    evaluate_agent_requirement_coverage(
                        llm=llm,
                        question=question,
                        requirements=requirements,
                        context=evidence_context,
                        valid_evidence_ids=valid_evidence_ids,
                    )
                )

                best_coverage_state = (
                    merge_monotonic_coverage(
                        previous_state=best_coverage_state,
                        new_state=new_coverage_state,
                    )
                )

                coverage_state = best_coverage_state

                coverage_complete = (
                    all_requirements_covered(
                        coverage_state
                    )
                )

                trace.append(
                    {
                        "step": step + 1,
                        "type": "coverage_check",
                        "requirements": requirements,
                        "coverage": coverage_state,
                        "complete": coverage_complete,
                        "evidence_ids": [
                            item["source_id"]
                            for item in evidence
                        ],
                    }
                )

                if not coverage_complete:

                    uncovered = (
                        get_uncovered_requirements(
                            coverage_state
                        )
                    )

                    missing_text = "\n".join(
                        (
                            f"- {item['id']} "
                            f"[{item['status']}]: "
                            f"{item['requirement']}"
                        )
                        for item in uncovered
                    )

                    follow_up_searches = (
                        plan_follow_up_searches(
                            llm=llm,
                            question=question,
                            coverage_state=coverage_state,
                            context=evidence_context,
                            previous_searches=previous_searches,
                        )
                    )

                    planned_search_text = "\n".join(
                        f"- {query}"
                        for query in follow_up_searches
                    )

                    messages.append(
                        HumanMessage(
                            content=(
                                "Your proposed answer cannot yet be "
                                "accepted because the retrieved document "
                                "evidence does not cover all requirements.\n\n"

                                "The following evidence requirements are "
                                "still PARTIAL or MISSING:\n"
                                f"{missing_text}\n\n"

                                "The retrieval planner suggests these "
                                "focused search directions:\n"
                                f"{planned_search_text}\n\n"

                                "Choose ONE focused search direction and "
                                "call search_document. Do not combine all "
                                "remaining requirements into one broad query. "
                                "Do not repeat a previous search. "
                                "Do not answer the original question yet."
                            )
                        )
                    )

                    continue

            # ------------------------------------------------
            # Validate whether the agent skipped calculator.
            # ------------------------------------------------

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
                    "arithmetic_required": (
                        validation["arithmetic_required"]
                    ),
                    "calculator_used": (
                        validation["calculator_used"]
                    ),
                    "retry_calculator": (
                        validation["retry_calculator"]
                    ),
                    "tools_used": list(tools_used),
                }
            )

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

                continue

            # ------------------------------------------------
            # Accept final response.
            # ------------------------------------------------

            trace.append(
                {
                    "step": step + 1,
                    "type": "final_answer",
                    "content": response.content,
                }
            )

            print(
                "ACCUMULATED AGENT EVIDENCE:",
                [
                    item["source_id"]
                    for item in agent_runtime.get_evidence()
                ],
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

            selected_tool = tool_registry.get(
                tool_name
            )

            # Values used by retrieval diagnostics.
            new_evidence_ids = set()
            stagnation_message = None

            if selected_tool is None:

                tool_result = {
                    "success": False,
                    "error": (
                        f"Unknown tool: {tool_name}"
                    ),
                }

            else:

                try:

                    # ----------------------------------------
                    # Snapshot evidence BEFORE retrieval.
                    # ----------------------------------------

                    evidence_ids_before = set()

                    if tool_name == "search_document":

                        evidence_ids_before = {
                            item["source_id"]
                            for item in (
                                agent_runtime.get_evidence()
                            )
                        }

                    # ----------------------------------------
                    # Execute tool.
                    # ----------------------------------------

                    tool_result = (
                        selected_tool.invoke(
                            tool_args
                        )
                    )

                    # ----------------------------------------
                    # Document-search bookkeeping.
                    # ----------------------------------------

                    if tool_name == "search_document":

                        search_query = (
                            tool_args.get("query")
                        )

                        if search_query:

                            previous_searches.append(
                                search_query
                            )

                        evidence_ids_after = {
                            item["source_id"]
                            for item in (
                                agent_runtime.get_evidence()
                            )
                        }

                        new_evidence_ids = (
                            evidence_ids_after
                            - evidence_ids_before
                        )

                        # ------------------------------------
                        # STAGNATION DETECTION
                        # ------------------------------------

                        if new_evidence_ids:

                            consecutive_stagnant_searches = 0

                        else:

                            consecutive_stagnant_searches += 1

                            uncovered_requirements = []

                            if best_coverage_state:

                                uncovered_requirements = (
                                    get_uncovered_requirements(
                                        best_coverage_state
                                    )
                                )

                            uncovered_text = "\n".join(
                                (
                                    f"- {item['id']} "
                                    f"[{item['status']}]: "
                                    f"{item['requirement']}"
                                )
                                for item
                                in uncovered_requirements
                            )

                            stagnation_message = (
                                "The previous document search "
                                "produced no new evidence. "
                                "All retrieved passages were "
                                "already present in the "
                                "accumulated evidence.\n\n"

                                "Do NOT repeat this search or "
                                "continue searching the same "
                                "requirement using only slightly "
                                "different wording.\n\n"

                                "The requirements that are still "
                                "PARTIAL or MISSING are:\n"
                                f"{uncovered_text}\n\n"

                                "Choose a DIFFERENT uncovered "
                                "requirement and perform one "
                                "focused search for evidence "
                                "supporting that requirement. "
                                "Do not answer the original "
                                "question yet."
                            )

                except Exception as exc:

                    tool_result = {
                        "success": False,
                        "error": str(exc),
                    }

            print("\nTOOL RESULT:")
            print(tool_result)

            if tool_name == "search_document":

                print(
                    "NEW EVIDENCE IDS:",
                    sorted(new_evidence_ids),
                )

                print(
                    "CONSECUTIVE STAGNANT SEARCHES:",
                    consecutive_stagnant_searches,
                )

            # ------------------------------------------------
            # Record exactly what happened.
            # ------------------------------------------------

            trace.append(
                {
                    "step": step + 1,
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                    "new_evidence_ids": (
                        sorted(new_evidence_ids)
                        if tool_name == "search_document"
                        else []
                    ),
                    "consecutive_stagnant_searches": (
                        consecutive_stagnant_searches
                        if tool_name == "search_document"
                        else 0
                    ),
                }
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # The ToolMessage must come immediately after
            # the AI's tool call. Only AFTER satisfying the
            # tool call do we inject the controller message.
            # ------------------------------------------------

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )

            if stagnation_message:

                messages.append(
                    HumanMessage(
                        content=stagnation_message
                    )
                )

    # --------------------------------------------------------
    # Agent reached its reasoning/retrieval budget.
    #
    # Do not discard useful accumulated evidence simply because
    # one or more semantic requirements remain PARTIAL/MISSING.
    # Produce the strongest grounded answer possible.
    # --------------------------------------------------------

    evidence = agent_runtime.get_evidence()

    if evidence:

        print(
            "\nAgent reached maximum number of steps. "
            "Generating grounded best-effort answer."
        )

        final_answer = generate_best_effort_answer(
            question=question,
            evidence=evidence,
            coverage_state=best_coverage_state,
        )

        coverage_complete = (
            all_requirements_covered(
                best_coverage_state
            )
            if best_coverage_state
            else False
        )

        trace.append(
            {
                "step": MAX_AGENT_STEPS,
                "type": "best_effort_synthesis",
                "content": final_answer,
                "coverage": best_coverage_state,
                "coverage_complete": coverage_complete,
                "evidence_ids": [
                    item["source_id"]
                    for item in evidence
                ],
                "termination_reason": "search_budget_exhausted",
            }
        )

        print("\nBEST-EFFORT FINAL ANSWER:")
        print(final_answer)

        return {
            "answer": final_answer,
            "trace": trace,
            "completed": True,
            "coverage_complete": coverage_complete,
            "termination_reason": "search_budget_exhausted",
        }


    # --------------------------------------------------------
    # No document evidence was gathered at all.
    #
    # In this situation grounded synthesis is impossible, so a
    # real failure is appropriate.
    # --------------------------------------------------------

    failure_message = (
        "I could not find enough document evidence "
        "to answer the request."
    )

    trace.append(
        {
            "step": MAX_AGENT_STEPS,
            "type": "max_steps",
            "content": failure_message,
        }
    )

    print(
        "\nAgent reached maximum number of steps "
        "without retrieving document evidence."
    )

    return {
        "answer": failure_message,
        "trace": trace,
        "completed": False,
        "coverage_complete": False,
        "termination_reason": "no_evidence",
    }