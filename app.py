import streamlit as st

from agent import run_agent
from agent_runtime import agent_runtime
from document_service import process_uploaded_pdf
from models import load_embeddings, load_llm
from rag import answer_question

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Local Agentic PDF RAG",
    page_icon="📄",
    layout="wide",
)

st.title("Local Agentic PDF RAG")

st.caption(
    "Upload a PDF and query it using a fully local RAG pipeline "
    "or an LLM agent with document tools."
)


# ============================================================
# Cached models
# ============================================================


@st.cache_resource
def get_embeddings():
    """
    Load the embedding model once and reuse it across Streamlit reruns.
    """
    return load_embeddings()


@st.cache_resource
def get_llm():
    """
    Load the Ollama LLM client once and reuse it across Streamlit reruns.
    """
    return load_llm()


embeddings = get_embeddings()
llm = get_llm()


# ============================================================
# Session state
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_info" not in st.session_state:
    st.session_state.document_info = None

if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_agent_answer" not in st.session_state:
    st.session_state.last_agent_answer = None


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    st.divider()

    mode = st.radio(
        "Mode",
        options=[
            "Verified RAG",
            "Agent",
        ],
        help=(
            "Verified RAG uses the existing retrieval, coverage, "
            "evidence-selection and verification pipeline. "
            "Agent mode lets the LLM choose tools dynamically."
        ),
    )

    st.divider()

    if st.session_state.document_info:

        info = st.session_state.document_info

        st.subheader("Loaded document")

        st.write(f"**Name:** {info.get('name', 'Unknown')}")
        st.write(f"**Pages:** {info.get('total_pages', 'Unknown')}")
        st.write(f"**Non-empty pages:** " f"{info.get('non_empty_pages', 'Unknown')}")
        st.write(f"**Chunks:** {info.get('chunks', 'Unknown')}")


# ============================================================
# Process uploaded document
# ============================================================

if uploaded_file is not None:

    # The name + size combination is sufficient for our current
    # single-document local application.
    current_file_key = (
        uploaded_file.name,
        uploaded_file.size,
    )

    if current_file_key != st.session_state.uploaded_file_key:

        with st.status(
            "Processing document...",
            expanded=True,
        ) as status:

            try:

                st.write("Reading PDF...")
                st.write("Chunking document...")
                st.write("Creating embeddings...")
                st.write("Building FAISS index...")

                vector_store, document_info = process_uploaded_pdf(
                    uploaded_file,
                    embeddings,
                )

                # ------------------------------------------------
                # Store document state for the normal RAG pipeline.
                # ------------------------------------------------

                st.session_state.vector_store = vector_store
                st.session_state.document_info = document_info
                st.session_state.uploaded_file_key = current_file_key

                # Clear previous answers because they belonged to
                # another document.
                st.session_state.last_result = None
                st.session_state.last_agent_answer = None

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Give the agent tools access to exactly the same
                # vector store and metadata used by the UI.
                # ------------------------------------------------

                agent_runtime.set_document(
                    vector_store=vector_store,
                    document_info=document_info,
                )

                status.update(
                    label="Document ready",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:

                status.update(
                    label="Document processing failed",
                    state="error",
                    expanded=True,
                )

                st.error(str(exc))
                st.stop()


# ============================================================
# Restore AgentRuntime after Streamlit reruns
# ============================================================

# Streamlit reruns this script whenever the user interacts with
# widgets. The vector store lives in session_state, so make sure
# AgentRuntime always points to the currently loaded document.

if st.session_state.vector_store is not None:

    agent_runtime.set_document(
        vector_store=st.session_state.vector_store,
        document_info=st.session_state.document_info,
    )


# ============================================================
# No document
# ============================================================

if st.session_state.vector_store is None:

    st.info("Upload a PDF from the sidebar to begin.")

    st.stop()


# ============================================================
# Document summary
# ============================================================

info = st.session_state.document_info

st.success(
    f"Loaded **{info.get('name', 'document')}** — "
    f"{info.get('total_pages', '?')} pages, "
    f"{info.get('chunks', '?')} indexed chunks."
)


# ============================================================
# VERIFIED RAG MODE
# ============================================================

if mode == "Verified RAG":

    st.header("Verified RAG")

    st.caption(
        "Uses evidence requirements, agentic retrieval, "
        "coverage tracking, evidence attribution, context "
        "budgeting and citation-scoped claim verification."
    )

    with st.form("rag_question_form"):

        question = st.text_area(
            "Ask a question about the document",
            placeholder=(
                "Example: What are the main ideas discussed " "in this document?"
            ),
            height=100,
        )

        submitted = st.form_submit_button(
            "Ask",
            type="primary",
        )

    if submitted:

        if not question.strip():

            st.warning("Enter a question first.")

        else:

            with st.status(
                "Running verified RAG pipeline...",
                expanded=True,
            ) as status:

                try:

                    st.write("Analyzing question...")
                    st.write("Retrieving candidate evidence...")
                    st.write("Evaluating evidence coverage...")
                    st.write("Selecting generation evidence...")
                    st.write("Generating grounded answer...")
                    st.write("Verifying claims...")

                    result = answer_question(
                        llm=llm,
                        vector_store=st.session_state.vector_store,
                        embeddings=embeddings,
                        question=question.strip(),
                    )

                    st.session_state.last_result = result

                    status.update(
                        label="Answer ready",
                        state="complete",
                        expanded=False,
                    )

                except Exception as exc:

                    status.update(
                        label="RAG pipeline failed",
                        state="error",
                        expanded=True,
                    )

                    st.error(str(exc))

    # --------------------------------------------------------
    # Render previous RAG result
    # --------------------------------------------------------

    result = st.session_state.last_result

    if result:

        st.subheader("Answer")

        st.markdown(
            result.get(
                "answer",
                "No answer was generated.",
            )
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = result.get("sources") or []

        if sources:

            with st.expander(
                "Sources",
                expanded=False,
            ):

                for source in sources:

                    source_id = source.get(
                        "source_id",
                        source.get("id", "Source"),
                    )

                    page = source.get("page")

                    text = source.get(
                        "text",
                        source.get("content", ""),
                    )

                    if page is not None:
                        st.markdown(f"**{source_id} — Page {page}**")
                    else:
                        st.markdown(f"**{source_id}**")

                    st.write(text)

                    st.divider()

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        with st.expander(
            "RAG diagnostics",
            expanded=False,
        ):

            requirements = result.get("requirements") or []

            coverage_state = result.get("coverage_state") or {}

            verified_claims = result.get("verified_claims") or []

            st.markdown("### Evidence requirements")

            if requirements:

                for requirement in requirements:

                    # Requirements may currently be strings or
                    # dictionaries depending on implementation.
                    if isinstance(requirement, dict):

                        requirement_id = requirement.get("id") or requirement.get(
                            "requirement_id",
                            "",
                        )

                        requirement_text = requirement.get("text") or requirement.get(
                            "requirement",
                            str(requirement),
                        )

                        label = (
                            f"{requirement_id}: " f"{requirement_text}"
                            if requirement_id
                            else requirement_text
                        )

                    else:

                        label = str(requirement)

                    st.write(f"- {label}")

            else:

                st.write("No requirements available.")

            st.markdown("### Coverage")

            if coverage_state:

                for key, value in coverage_state.items():

                    # Handle either a simple status string or
                    # a richer coverage object.
                    if isinstance(value, dict):
                        status_value = value.get(
                            "status",
                            str(value),
                        )
                    else:
                        status_value = str(value)

                    normalized = status_value.upper()

                    if normalized == "COVERED":
                        icon = "✅"
                    elif normalized == "PARTIAL":
                        icon = "🟡"
                    elif normalized == "MISSING":
                        icon = "❌"
                    else:
                        icon = "•"

                    st.write(f"{icon} **{key}** — " f"{status_value}")

            else:

                st.write("No coverage data available.")

            st.markdown("### Verified claims")

            if verified_claims:

                for claim in verified_claims:

                    if isinstance(claim, dict):

                        claim_text = claim.get(
                            "claim",
                            str(claim),
                        )

                        claim_sources = claim.get(
                            "sources",
                            "",
                        )

                        if claim_sources:

                            st.write(f"- {claim_text} " f"({claim_sources})")

                        else:

                            st.write(f"- {claim_text}")

                    else:

                        st.write(f"- {claim}")

            else:

                st.write("No verified claims available.")


# ============================================================
# AGENT MODE
# ============================================================

elif mode == "Agent":

    st.header("Tool-Calling Agent")

    st.caption(
        "The LLM dynamically chooses between document search, "
        "document metadata and calculator tools."
    )

    st.markdown("""
Available tools:

- **search_document** — semantic search over the uploaded PDF
- **document_info** — metadata about the uploaded PDF
- **calculator** — safe deterministic arithmetic
""")

    with st.form("agent_question_form"):

        agent_question = st.text_area(
            "Ask the agent",
            placeholder=(
                "Example: How many indexed chunks does this " "document have?"
            ),
            height=100,
        )

        agent_submitted = st.form_submit_button(
            "Run agent",
            type="primary",
        )

    if agent_submitted:

        if not agent_question.strip():

            st.warning("Enter a question first.")

        else:

            with st.status(
                "Agent is working...",
                expanded=True,
            ) as status:

                try:

                    st.write("Giving the agent access to tools...")

                    agent_result = run_agent(agent_question.strip())

                    st.session_state.last_agent_answer = agent_result

                    status.update(
                        label="Agent finished",
                        state="complete",
                        expanded=False,
                    )

                except Exception as exc:

                    status.update(
                        label="Agent failed",
                        state="error",
                        expanded=True,
                    )

                    st.error(str(exc))

    # --------------------------------------------------------
    # Render previous agent answer
    # --------------------------------------------------------

    if st.session_state.last_agent_answer:

        agent_result = st.session_state.last_agent_answer

        # ========================================================
        # Final answer
        # ========================================================

        st.subheader("Agent answer")

        st.markdown(
            agent_result.get(
                "answer",
                "No answer was generated.",
            )
        )

        # ========================================================
        # Agent execution trace
        # ========================================================

        trace = agent_result.get("trace", [])

        with st.expander(
            "Agent trace",
            expanded=False,
        ):

            if not trace:

                st.write("No trace information available.")

            else:

                for event in trace:

                    step = event.get("step")
                    event_type = event.get("type")

                    # --------------------------------------------
                    # Tool call
                    # --------------------------------------------

                    if event_type == "tool_call":

                        tool_name = event.get(
                            "tool",
                            "unknown",
                        )

                        arguments = event.get(
                            "arguments",
                            {},
                        )

                        result = event.get(
                            "result",
                            {},
                        )

                        new_evidence_ids = event.get(
                            "new_evidence_ids",
                            [],
                        )

                        stagnant_searches = event.get(
                            "consecutive_stagnant_searches",
                            0,
                        )

                        st.markdown(f"### Step {step} — `{tool_name}`")

                        st.markdown("**Arguments**")

                        st.json(arguments)

                        st.markdown("**Result**")

                        st.json(result)

                        if tool_name == "search_document":

                            st.markdown(
                                "**Retrieval diagnostics**"
                            )

                            if new_evidence_ids:

                                st.write(
                                    "New evidence: "
                                    + ", ".join(
                                        new_evidence_ids
                                    )
                                )

                            else:

                                st.write(
                                    "New evidence: None"
                                )

                            st.write(
                                "Consecutive stagnant searches: "
                                f"{stagnant_searches}"
                            )

                    # --------------------------------------------
                    # Evidence coverage check
                    # --------------------------------------------

                    elif event_type == "coverage_check":

                        st.markdown(
                            f"### Step {step} — Evidence coverage"
                        )

                        requirements = event.get(
                            "requirements",
                            [],
                        )

                        coverage = event.get(
                            "coverage",
                            {},
                        )

                        complete = event.get(
                            "complete",
                            False,
                        )

                        evidence_ids = event.get(
                            "evidence_ids",
                            [],
                        )

                        # ----------------------------------------
                        # Requirements
                        # ----------------------------------------

                        st.markdown("**Requirements**")

                        if requirements:

                            for requirement in requirements:

                                if isinstance(requirement, dict):

                                    requirement_id = requirement.get(
                                        "id",
                                        requirement.get(
                                            "requirement_id",
                                            "",
                                        ),
                                    )

                                    requirement_text = requirement.get(
                                        "text",
                                        requirement.get(
                                            "requirement",
                                            str(requirement),
                                        ),
                                    )

                                    if requirement_id:
                                        st.write(
                                            f"- **{requirement_id}**: "
                                            f"{requirement_text}"
                                        )
                                    else:
                                        st.write(
                                            f"- {requirement_text}"
                                        )

                                else:

                                    st.write(
                                        f"- {requirement}"
                                    )

                        else:

                            st.write(
                                "No evidence requirements available."
                            )

                        # ----------------------------------------
                        # Coverage state
                        # ----------------------------------------

                        st.markdown("**Coverage state**")

                        if coverage:

                            for item in coverage:

                                if isinstance(item, dict):

                                    requirement_id = item.get(
                                        "id",
                                        item.get(
                                            "requirement_id",
                                            "?",
                                        ),
                                    )

                                    requirement_text = item.get(
                                        "requirement",
                                        item.get(
                                            "text",
                                            "",
                                        ),
                                    )

                                    status_value = item.get(
                                        "status",
                                        "UNKNOWN",
                                    )

                                    item_evidence_ids = item.get(
                                        "evidence_ids",
                                        [],
                                    )

                                    reason = item.get(
                                        "reason",
                                        "",
                                    )

                                else:

                                    requirement_id = "?"
                                    requirement_text = ""
                                    status_value = str(item)
                                    item_evidence_ids = []
                                    reason = ""

                                normalized = status_value.upper()

                                if normalized == "COVERED":
                                    icon = "✅"
                                elif normalized == "PARTIAL":
                                    icon = "🟡"
                                elif normalized == "MISSING":
                                    icon = "❌"
                                else:
                                    icon = "•"

                                st.write(
                                    f"{icon} **{requirement_id}** — "
                                    f"{status_value}"
                                )

                                if requirement_text:
                                    st.caption(requirement_text)

                                if item_evidence_ids:
                                    st.caption(
                                        "Evidence: "
                                        + ", ".join(
                                            item_evidence_ids
                                        )
                                    )

                                if reason:
                                    st.caption(
                                        f"Reason: {reason}"
                                    )

                        else:

                            st.write(
                                "No coverage state available."
                            )

                        # ----------------------------------------
                        # Evidence accumulated so far
                        # ----------------------------------------

                        st.markdown("**Accumulated evidence**")

                        if evidence_ids:
                            st.write(
                                ", ".join(evidence_ids)
                            )
                        else:
                            st.write(
                                "No document evidence accumulated."
                            )

                        # ----------------------------------------
                        # Completion decision
                        # ----------------------------------------

                        if complete:
                            st.success(
                                "All evidence requirements are covered."
                            )
                        else:
                            st.warning(
                                "Evidence coverage is incomplete. "
                                "The agent must continue searching."
                            )

                    # --------------------------------------------
                    # Validation retry
                    # --------------------------------------------

                    elif event_type == "validation_retry":
                    
                        st.markdown(
                            f"### Step {step} — Validation retry"
                        )

                        st.warning(
                            event.get(
                                "reason",
                                "Agent response failed validation.",
                            )
                        )
                    elif event_type == "validation_debug":

                        st.markdown(
                            f"### Step {step} — Validator"
                        )

                        st.json(
                            {
                                "decision": event.get("decision"),
                                "arithmetic_required": event.get(
                                    "arithmetic_required"
                                ),
                                "calculator_used": event.get(
                                    "calculator_used"
                                ),
                                "retry_calculator": event.get(
                                    "retry_calculator"
                                ),
                                "tools_used": event.get(
                                    "tools_used",
                                    [],
                                ),
                            }
                        )
                    # --------------------------------------------
                    # Final answer
                    # --------------------------------------------

                    elif event_type == "final_answer":

                        st.markdown(f"### Step {step} — Final answer")

                        st.write(event.get("content", ""))

                    # --------------------------------------------
                    # Max-step safety stop
                    # --------------------------------------------

                    elif event_type == "max_steps":

                        st.markdown(f"### Step {step} — Safety stop")

                        st.warning(event.get("content", ""))

                    elif event_type == "best_effort_synthesis":

                        st.markdown(
                            f"### Step {event['step']} — "
                            "Grounded synthesis"
                        )

                        st.caption(
                            "The retrieval budget was exhausted. "
                            "The agent generated the final answer from "
                            "the accumulated document evidence."
                        )

                        st.write(event["content"])

                        evidence_ids = event.get(
                            "evidence_ids",
                            [],
                        )

                        if evidence_ids:

                            st.caption(
                                "Evidence used: "
                                + ", ".join(evidence_ids)
                            )

                    
                    st.divider()

    # ========================================================
    # Completion state
    # ========================================================

        if not agent_result.get("completed", False):

            st.warning("The agent stopped before completing " "the request.")

        st.info(
            "Agent mode is experimental. Tool selection and "
            "planning are performed by the local LLM and may "
            "not always follow the optimal tool sequence."
        )
