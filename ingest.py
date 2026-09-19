from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models import load_embeddings
from config import FAISS_INDEX_PATH

PDF_PATH = "data/book.pdf"


def main():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Loaded {len(pages)} PDF pages.")

    pages = [page for page in pages if page.page_content.strip()]
    print(f"{len(pages)} non-empty pages remain.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks.")

    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"Saved FAISS index to: {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
