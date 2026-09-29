import os
import tempfile
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# DOCUMENT SETTINGS
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# SAVE UPLOADED PDF TEMPORARILY
# ============================================================

def save_uploaded_pdf(uploaded_file):
    """
    Save a Streamlit UploadedFile to a temporary PDF.

    PyPDFLoader requires a filesystem path.
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        return temp_file.name


def create_document_id(display_name):
    """
    Create a stable document identifier from the filename.

    Example:

        "NoSQL Distilled.pdf"
            ->
        "nosql-distilled"
    """

    name_without_extension = os.path.splitext(
        display_name
    )[0]

    document_id = re.sub(
        r"[^a-zA-Z0-9]+",
        "-",
        name_without_extension,
    )

    return document_id.strip("-").lower()

# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(
    pdf_path,
    display_name=None,
):
    """
    Load PDF pages and remove pages containing no text.

    display_name is the human-readable filename that should be
    shown later in the UI.

    Example:

        Effective Java.pdf

    rather than:

        C:\\Users\\...\\Temp\\tmp1234.pdf
    """

    loader = PyPDFLoader(
        pdf_path
    )

    pages = loader.load()

    # --------------------------------------------------------
    # Attach UI-friendly metadata BEFORE chunking.
    #
    # LangChain's splitter will preserve this metadata on the
    # chunks it creates.
    # --------------------------------------------------------

    if display_name:

        document_id = create_document_id(
            display_name
        )

        for page in pages:

            page.metadata[
                "display_name"
            ] = display_name

            page.metadata[
                "document_id"
            ] = document_id

    non_empty_pages = [
        page
        for page in pages
        if page.page_content.strip()
    ]

    return (
        pages,
        non_empty_pages,
    )


# ============================================================
# CHUNK DOCUMENT
# ============================================================

def chunk_pages(pages):
    """
    Split pages into overlapping chunks.

    Page metadata is preserved automatically.
    Each chunk is assigned a stable ID for the current
    ingestion configuration.
    """

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    )

    chunks = text_splitter.split_documents(
        pages
    )

    # Assign a deterministic ID to every chunk.
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index

    return chunks


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store(
    chunks,
    embeddings,
):
    """
    Build an in-memory FAISS vector store.
    """

    if not chunks:

        raise ValueError(
            "Cannot build vector store: "
            "the PDF produced no text chunks."
        )

    return FAISS.from_documents(
        chunks,
        embeddings,
    )


def process_pdf_to_chunks(
    pdf_path,
    display_name=None,
):
    """
    Load and chunk one PDF without building a vector store.

    This allows chunks from multiple PDFs to be combined
    before creating a shared FAISS index.
    """

    (
        all_pages,
        non_empty_pages,
    ) = load_pdf(
        pdf_path,
        display_name,
    )

    chunks = chunk_pages(
        non_empty_pages
    )

    document_info = {
        "name": (
            display_name
            or os.path.basename(pdf_path)
        ),
        "document_id": (
            create_document_id(
                display_name
                or os.path.basename(pdf_path)
            )
        ),
        "total_pages": len(all_pages),
        "non_empty_pages": len(
            non_empty_pages
        ),
        "chunks": len(chunks),
    }

    return chunks, document_info

# ============================================================
# COMPLETE DOCUMENT PIPELINE
# ============================================================

def process_pdf(
    pdf_path,
    embeddings,
    display_name=None,
):
    """
    Process one PDF into an in-memory FAISS index.
    """

    (
        chunks,
        document_info,
    ) = process_pdf_to_chunks(
        pdf_path,
        display_name,
    )

    vector_store = build_vector_store(
        chunks,
        embeddings,
    )

    return (
        vector_store,
        document_info,
    )

# ============================================================
# STREAMLIT UPLOAD PIPELINE
# ============================================================

def process_uploaded_pdf(
    uploaded_file,
    embeddings,
):
    """
    Process a Streamlit UploadedFile.

    The temporary PDF is deleted after FAISS has been built,
    but the original filename remains stored in chunk metadata.
    """

    temp_path = None

    try:

        temp_path = (
            save_uploaded_pdf(
                uploaded_file
            )
        )

        return process_pdf(
            temp_path,
            embeddings,
            display_name=(
                uploaded_file.name
            ),
        )

    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )



def process_uploaded_pdfs(
    uploaded_files,
    embeddings,
):
    """
    Process multiple Streamlit UploadedFiles into one
    shared FAISS vector store.
    """

    all_chunks = []
    documents_info = []

    for uploaded_file in uploaded_files:

        temp_path = None

        try:

            temp_path = save_uploaded_pdf(
                uploaded_file
            )

            (
                chunks,
                document_info,
            ) = process_pdf_to_chunks(
                temp_path,
                display_name=uploaded_file.name,
            )

            all_chunks.extend(chunks)

            documents_info.append(
                document_info
            )

        finally:

            if (
                temp_path
                and os.path.exists(temp_path)
            ):
                os.remove(temp_path)

    vector_store = build_vector_store(
        all_chunks,
        embeddings,
    )

    return (
        vector_store,
        documents_info,
    )

# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    from models import load_embeddings

    print(
        "Loading embedding model..."
    )

    embeddings = (
        load_embeddings()
    )

    print(
        "Processing data/book.pdf..."
    )

    (
        vector_store,
        document_info,
    ) = process_pdf(
        "data/book.pdf",
        embeddings,
        display_name="NoSQL Distilled",
    )

    print()

    print(
        "=========================================="
    )

    print(
        "DOCUMENT INGESTION TEST"
    )

    print(
        "=========================================="
    )

    print(
        f"Document: "
        f"{document_info['name']}"
    )

    print(
        f"Total PDF pages: "
        f"{document_info['total_pages']}"
    )

    print(
        f"Non-empty pages: "
        f"{document_info['non_empty_pages']}"
    )

    print(
        f"Chunks created: "
        f"{document_info['chunks']}"
    )

    print()

    print(
        "FAISS vector store created successfully. ✅"
    )