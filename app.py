import time

import streamlit as st

from models import (
    load_embeddings,
    load_llm,
)

from document_service import (
    process_uploaded_pdf,
)

from rag import answer_question


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Local Agentic PDF RAG",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def get_embeddings():
    """
    Load embeddings only once per Streamlit process.
    """

    return load_embeddings()


@st.cache_resource
def get_llm():
    """
    Create the Ollama client only once.
    """

    return load_llm()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📚 Local Agentic PDF RAG"
)

st.caption(
    "Query PDFs using local embeddings, FAISS, "
    "agentic retrieval, evidence attribution, "
    "and a locally hosted LLM."
)

st.divider()


# ============================================================
# LOAD LOCAL MODELS
# ============================================================

with st.spinner(
    "Loading local AI models..."
):

    embeddings = get_embeddings()

    llm = get_llm()


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

if uploaded_file is not None:

    current_file_key = (
        uploaded_file.name,
        uploaded_file.size,
    )

    previous_file_key = (
        st.session_state.get(
            "file_key"
        )
    )

    # --------------------------------------------------------
    # Only rebuild FAISS when the uploaded document changes.
    # --------------------------------------------------------

    if (
        previous_file_key
        != current_file_key
    ):

        with st.spinner(
            "Reading, chunking, embedding, "
            "and indexing PDF..."
        ):

            try:

                (
                    vector_store,
                    document_info,
                ) = process_uploaded_pdf(
                    uploaded_file,
                    embeddings,
                )

                st.session_state[
                    "vector_store"
                ] = vector_store

                st.session_state[
                    "document_info"
                ] = document_info

                st.session_state[
                    "file_key"
                ] = current_file_key

                # Clear answer from previous document.
                st.session_state.pop(
                    "rag_result",
                    None,
                )

                st.session_state.pop(
                    "rag_elapsed",
                    None,
                )

                st.session_state.pop(
                    "last_question",
                    None,
                )

            except Exception as error:

                st.error(
                    "Could not process the PDF."
                )

                st.exception(error)

                st.stop()


    # ========================================================
    # DOCUMENT STATE
    # ========================================================

    document_info = (
        st.session_state[
            "document_info"
        ]
    )

    vector_store = (
        st.session_state[
            "vector_store"
        ]
    )


    # ========================================================
    # DOCUMENT HEADER
    # ========================================================

    st.success(
        "PDF indexed successfully."
    )

    st.subheader(
        f"📄 {document_info['name']}"
    )

    column1, column2, column3 = (
        st.columns(3)
    )

    with column1:

        st.metric(
            "PDF Pages",
            document_info[
                "total_pages"
            ],
        )

    with column2:

        st.metric(
            "Pages With Text",
            document_info[
                "non_empty_pages"
            ],
        )

    with column3:

        st.metric(
            "Chunks",
            document_info[
                "chunks"
            ],
        )

    st.divider()


    # ========================================================
    # QUESTION FORM
    # ========================================================

    st.subheader(
        "Ask your document"
    )

    # Using a form means pressing Enter or clicking Ask submits
    # the question cleanly as one action.

    with st.form(
        "question_form"
    ):

        question = st.text_input(
            "Question",
            placeholder=(
                "Ask something about the uploaded PDF..."
            ),
        )

        ask_button = (
            st.form_submit_button(
                "Ask",
                type="primary",
            )
        )


    # ========================================================
    # RUN RAG
    # ========================================================

    if ask_button:

        question = (
            question.strip()
        )

        if not question:

            st.warning(
                "Enter a question first."
            )

        else:

            start_time = (
                time.perf_counter()
            )

            try:

                with st.status(
                    "Running agentic RAG...",
                    expanded=True,
                ) as status:

                    st.write(
                        "🔎 Retrieving and evaluating evidence..."
                    )

                    st.write(
                        "🧠 Planning follow-up searches when needed..."
                    )

                    st.write(
                        "📚 Attributing evidence to requirements..."
                    )

                    st.write(
                        "✍️ Generating and verifying the answer..."
                    )

                    rag_result = (
                        answer_question(
                            llm,
                            vector_store,
                            embeddings,
                            question,
                        )
                    )

                    status.update(
                        label=(
                            "RAG pipeline complete"
                        ),
                        state="complete",
                        expanded=False,
                    )

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                st.session_state[
                    "rag_result"
                ] = rag_result

                st.session_state[
                    "rag_elapsed"
                ] = elapsed

                st.session_state[
                    "last_question"
                ] = question

            except Exception as error:

                st.error(
                    "An error occurred while "
                    "answering the question."
                )

                st.exception(error)


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    rag_result = (
        st.session_state.get(
            "rag_result"
        )
    )

    if rag_result:

        st.divider()

        last_question = (
            st.session_state.get(
                "last_question"
            )
        )

        if last_question:

            st.caption(
                "QUESTION"
            )

            st.markdown(
                f"**{last_question}**"
            )


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader(
            "Answer"
        )

        st.markdown(
            rag_result["answer"]
        )

        elapsed = (
            st.session_state.get(
                "rag_elapsed"
            )
        )

        if elapsed is not None:

            st.caption(
                f"⏱ Answered in "
                f"{elapsed:.1f} seconds"
            )


        # ====================================================
        # SOURCES
        # ====================================================

        sources = (
            rag_result.get(
                "sources",
                [],
            )
        )

        if sources:

            with st.expander(
                f"Sources ({len(sources)})"
            ):

                for source in sources:

                    source_id = (
                        source.get(
                            "id",
                            "S?",
                        )
                    )

                    document_name = (
                        source.get(
                            "document",
                            "Uploaded PDF",
                        )
                    )

                    page = (
                        source.get(
                            "page",
                            "?",
                        )
                    )

                    st.markdown(
                        f"**[{source_id}]** "
                        f"{document_name} "
                        f"— Page {page}"
                    )


        # ====================================================
        # RAG DIAGNOSTICS
        # ====================================================

        with st.expander(
            "RAG diagnostics"
        ):

            # -----------------------------------------------
            # REQUIREMENTS
            # -----------------------------------------------

            st.markdown(
                "### Evidence requirements"
            )

            requirements = (
                rag_result.get(
                    "requirements",
                    [],
                )
            )

            for requirement in (
                requirements
            ):

                st.markdown(
                    f"**{requirement['id']}** — "
                    f"{requirement['requirement']}"
                )


            # -----------------------------------------------
            # COVERAGE
            # -----------------------------------------------

            st.markdown(
                "### Final coverage"
            )

            coverage_state = (
                rag_result.get(
                    "coverage_state",
                    [],
                )
            )

            for item in (
                coverage_state
            ):

                status = (
                    item["status"]
                )

                if status == "COVERED":

                    icon = "✅"

                elif status == "PARTIAL":

                    icon = "🟡"

                else:

                    icon = "❌"

                st.markdown(
                    f"{icon} "
                    f"**{item['id']} — "
                    f"{status}**"
                )

                st.caption(
                    item[
                        "requirement"
                    ]
                )


            # -----------------------------------------------
            # VERIFIED CLAIMS
            # -----------------------------------------------

            st.markdown(
                "### Verified claims"
            )

            verified_claims = (
                rag_result.get(
                    "verified_claims",
                    [],
                )
            )

            if verified_claims:

                for claim in (
                    verified_claims
                ):

                    st.markdown(
                        f"- {claim['claim']} "
                        f"`[{claim['sources']}]`"
                    )

            else:

                st.write(
                    "No verified claims."
                )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "Upload a PDF to begin."
    )