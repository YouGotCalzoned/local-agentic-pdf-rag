from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from config import EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME, FAISS_INDEX_PATH


def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_vector_store(embeddings):
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_llm():
    return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0,  num_ctx=16384,)
