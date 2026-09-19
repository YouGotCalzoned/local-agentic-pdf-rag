import os
import tempfile

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

        for page in pages:

            page.metadata[
                "display_name"
            ] = display_name

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
    """

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    )

    return (
        text_splitter.split_documents(
            pages
        )
    )


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


# ============================================================
# COMPLETE DOCUMENT PIPELINE
# ============================================================

def process_pdf(
    pdf_path,
    embeddings,
    display_name=None,
):
    """
    Process a PDF into an in-memory FAISS index.

    PDF
     ↓
    pages
     ↓
    remove empty pages
     ↓
    chunks
     ↓
    embeddings
     ↓
    FAISS
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

    vector_store = (
        build_vector_store(
            chunks,
            embeddings,
        )
    )

    document_info = {
        "name": (
            display_name
            or os.path.basename(
                pdf_path
            )
        ),
        "total_pages": len(
            all_pages
        ),
        "non_empty_pages": len(
            non_empty_pages
        ),
        "chunks": len(
            chunks
        ),
    }

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