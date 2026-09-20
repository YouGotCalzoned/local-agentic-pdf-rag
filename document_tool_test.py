from agent_runtime import agent_runtime
from document_service import process_pdf
from models import load_embeddings

from tools.document_search import search_document


PDF_PATH = "data/book.pdf"


print("Loading embeddings...")
embeddings = load_embeddings()

print("Processing document...")
vector_store, document_info = process_pdf(
    PDF_PATH,
    embeddings,
)

agent_runtime.set_document(
    vector_store=vector_store,
    document_info=document_info,
)


print("\nDOCUMENT INFO:")
print(document_info)


result = search_document.invoke(
    {
        "query": "different types of NoSQL databases"
    }
)


print("\nSEARCH RESULT:")

for passage in result["passages"]:

    print("\n" + "=" * 60)
    print(
        f'{passage["passage_id"]} | '
        f'Page {passage["page"]} | '
        f'Distance {passage["distance"]:.4f}'
    )
    print("=" * 60)

    print(passage["text"][:700])