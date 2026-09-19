# Local Agentic PDF RAG

A fully local Retrieval-Augmented Generation (RAG) application for querying arbitrary PDF documents using local embeddings, FAISS vector search, and a locally hosted LLM.

Unlike a basic "retrieve top-k chunks and send them to an LLM" pipeline, this project adds agentic retrieval, evidence requirement decomposition, coverage tracking, evidence attribution, context budgeting, and citation-scoped claim verification.

The application runs locally and does not require a cloud LLM API.

---

## Features

- Upload and query arbitrary PDF documents through a Streamlit interface
- Local sentence-transformer embeddings
- FAISS vector search
- Local Llama 3.1 8B inference through Ollama
- Semantic relevance gating for out-of-domain questions
- MMR-based retrieval for balancing relevance and diversity
- LLM-generated evidence requirements
- Agentic follow-up retrieval for missing evidence
- Requirement-level coverage tracking
- Evidence attribution before generation
- Deterministic context budgeting
- Source-grounded answer generation
- Citation-scoped claim verification
- Page-level source references
- RAG diagnostics exposed in the UI

---

## Architecture

```text
                        User
                          │
                          ▼
                   Streamlit UI
                          │
              ┌───────────┴───────────┐
              │                       │
          Upload PDF              Ask Question
              │                       │
              ▼                       ▼
        PyPDFLoader            Evidence Requirements
              │                       │
              ▼                       ▼
          Chunking              Initial Retrieval
              │                       │
              ▼                       ▼
      MiniLM Embeddings         Coverage Check
              │                       │
              ▼                       ▼
            FAISS ◄────── Follow-up Retrieval
                                      │
                                      ▼
                              Evidence Attribution
                                      │
                                      ▼
                              Context Budgeting
                                + MMR Selection
                                      │
                                      ▼
                                Llama 3.1 8B
                                      │
                                      ▼
                               Atomic Claims
                                      │
                                      ▼
                         Citation-Scoped Verification
                                      │
                                      ▼
                         Verified Answer + Sources
```

The pipeline deliberately separates **retrieval**, **reasoning**, **evidence selection**, **generation**, and **verification** instead of treating RAG as a single LLM call.

---

## How It Works

### 1. PDF ingestion

Uploaded PDFs are:

1. Parsed using `PyPDFLoader`
2. Filtered to remove empty pages
3. Split into overlapping text chunks
4. Embedded using `sentence-transformers/all-MiniLM-L6-v2`
5. Indexed in an in-memory FAISS vector store

The current chunking configuration uses:

```text
Chunk size:    1000 characters
Chunk overlap: 200 characters
```

---

### 2. Evidence requirement decomposition

Before retrieval, the LLM decomposes the user's question into a small set of evidence requirements.

For example:

```text
Question:
How are replication, consistency, and availability related
in distributed databases?

Possible requirements:

R1: Explain replication and consistency
R2: Explain how replication affects availability
R3: Explain the trade-offs between consistency and availability
```

These requirements become explicit retrieval goals rather than relying entirely on a single similarity search.

---

### 3. Initial retrieval

The question is embedded and searched against FAISS.

The initial retrieval stage combines:

- vector similarity
- semantic relevance gating
- Maximal Marginal Relevance (MMR)

The relevance gate can reject questions whose retrieved chunks are too distant from the document corpus.

This allows clearly out-of-domain questions to terminate early instead of forcing the LLM to invent an answer.

---

### 4. Agentic retrieval

After initial retrieval, the LLM evaluates whether the current evidence satisfies each requirement.

Each requirement is classified as:

```text
COVERED
PARTIAL
MISSING
```

If evidence is incomplete, the LLM proposes targeted follow-up searches.

Python controls:

- retrieval rounds
- maximum follow-up queries
- search history
- deduplication
- stopping conditions

The LLM therefore performs planning, while deterministic application code controls execution.

```text
Question
   ↓
Initial retrieval
   ↓
Coverage evaluation
   ↓
Enough evidence? ───── Yes ────→ Continue
   │
   No
   ↓
Generate targeted searches
   ↓
Retrieve additional evidence
   ↓
Re-evaluate coverage
```

This loop is bounded to prevent uncontrolled retrieval.

---

## Evidence Attribution

Retrieving a semantically similar chunk does not necessarily mean that the chunk actually supports an answer.

After agentic retrieval, the system performs a separate evidence-attribution stage.

The LLM evaluates which retrieved passages actually support each evidence requirement.

This creates a distinction between:

```text
Embedding similarity
        ↓
"Could this passage be relevant?"

Evidence attribution
        ↓
"Does this passage actually support this requirement?"
```

This helps prevent semantically related but low-value chunks—such as index pages, references, or tangential discussions—from dominating the generation context.

---

## Context Budgeting

Agentic retrieval may accumulate many candidate passages.

Sending every retrieved chunk to the generator can cause:

- context-window pressure
- redundant evidence
- irrelevant context
- source confusion
- citation mistakes

The application therefore builds a smaller generation context.

Current limits:

```text
Maximum generation chunks: 8
Maximum context characters: 10,000
```

Requirement-supporting evidence is prioritized first.

Remaining capacity can be filled using MMR to preserve useful diversity.

This creates two distinct optimization stages:

```text
Retrieval
    ↓
Optimize recall

Evidence composition
    ↓
Optimize the final LLM prompt
```

---

## Grounded Generation

The generator receives:

- the original user question
- a bounded set of selected evidence
- labeled source passages

Sources are represented internally using IDs such as:

```text
[S1]
[S2]
[S3]
```

The model is instructed to answer only from retrieved evidence and attach source IDs to factual claims.

---

## Citation-Scoped Claim Verification

The first generated answer is treated as a draft rather than automatically trusted.

The system:

1. Extracts atomic factual claims from the draft
2. Extracts the source IDs cited by each claim
3. Resolves those IDs to the exact retrieved passages
4. Gives the verifier only the claim and its cited evidence
5. Classifies the claim as `SUPPORTED` or `UNSUPPORTED`
6. Removes unsupported claims
7. Produces the final answer from verified claims

```text
Draft Answer
     ↓
Atomic Claims
     ↓
Claim + cited source IDs
     ↓
Python resolves exact passages
     ↓
Verifier
     ↓
SUPPORTED / UNSUPPORTED
     ↓
Verified Claims
     ↓
Final Answer
```

The verifier cannot silently substitute another source because citation resolution is performed deterministically by Python.

---

## Why Citation-Scoped Verification?

An earlier implementation verified claims against the entire retrieved context.

That created false positives and false negatives because the verifier could use evidence unrelated to the source cited by the generator.

The current architecture verifies:

```text
Claim C1
   ↓
Cited sources: [S2], [S5]
   ↓
Verifier sees ONLY S2 + S5
```

This makes verification stricter and easier to reason about.

---

## Local Models

### Embeddings

```text
sentence-transformers/all-MiniLM-L6-v2
```

Used for document and query embeddings.

### LLM

```text
Llama 3.1 8B
```

served locally through Ollama.

The current Ollama context window is configured to:

```text
16384 tokens
```

No cloud LLM API is required.

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF parsing | PyPDFLoader / pypdf |
| Text splitting | LangChain text splitters |
| Embeddings | Sentence Transformers |
| Vector database | FAISS |
| LLM runtime | Ollama |
| LLM | Llama 3.1 8B |
| RAG orchestration | Python |
| Verification | LLM + deterministic Python citation resolution |

---

## Project Structure

```text
rag-learning/
│
├── app.py
├── config.py
├── context.py
├── coverage.py
├── document_service.py
├── evidence.py
├── generation.py
├── ingest.py
├── mmr_test.py
├── models.py
├── planner.py
├── rag.py
├── retrieval.py
├── verification.py
├── requirements.txt
└── .gitignore
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI |
| `config.py` | Retrieval, model, context, and pipeline configuration |
| `context.py` | Context construction and source labeling |
| `coverage.py` | Requirement-to-evidence attribution |
| `document_service.py` | Arbitrary PDF ingestion and indexing |
| `evidence.py` | Generation-time evidence selection and context budgeting |
| `generation.py` | Draft and final answer generation |
| `ingest.py` | Original fixed-document ingestion pipeline |
| `models.py` | Embedding model, FAISS, and Ollama initialization |
| `planner.py` | Evidence requirements, coverage evaluation, and retrieval planning |
| `rag.py` | Main RAG orchestration |
| `retrieval.py` | Similarity search, MMR, filtering, and deduplication |
| `verification.py` | Atomic claim extraction and citation-scoped verification |

---

## Installation

### Requirements

You will need:

- Python 3
- Ollama
- Git

Clone the repository:

```bash
git clone <repository-url>
cd local-agentic-pdf-rag
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Ollama Setup

Install Ollama and pull Llama 3.1 8B:

```bash
ollama pull llama3.1:8b
```

Verify that the model is available:

```bash
ollama list
```

Ollama must be running when the application sends requests to the model.

---

## Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit in your browser.

Then:

1. Upload a PDF
2. Wait for the document to be parsed, chunked, embedded, and indexed
3. Enter a question
4. Run the RAG pipeline
5. Inspect the answer, sources, evidence coverage, and verification diagnostics

---

## Example Use Cases

The same pipeline can operate across different document domains without domain-specific retrieval code.

Examples tested during development include:

### Technical books

```text
What are the different types of NoSQL databases?

How are graph databases different from other NoSQL databases?

How are replication, consistency, and availability related
in distributed databases?
```

### Programming books

```text
What are the best practices for writing Java code?

How should exceptions be handled?
```

### Running / training books

```text
How can injuries be prevented while training for a marathon?
```

---

## Out-of-Domain Handling

The application uses a configurable FAISS distance threshold before entering the more expensive agentic pipeline.

If the initial retrieval results are too distant from the query, the application can terminate early instead of asking the LLM to construct an answer from irrelevant context.

For example, a question unrelated to the uploaded document should fail the relevance gate rather than encourage hallucination.

The current threshold was determined experimentally and is specific to the embedding model, corpus characteristics, distance metric, and chunking strategy.

---

## Design Principles

### LLMs reason; code controls

LLMs are used where semantic judgment is useful:

- decomposing questions
- evaluating evidence coverage
- planning follow-up searches
- attributing evidence
- generating answers
- verifying claims

Python handles deterministic control:

- loop limits
- retrieval execution
- thresholds
- source-ID resolution
- deduplication
- context budgets
- stopping conditions

---

### Retrieval and generation are separate problems

A relevant passage being present in the candidate pool does not guarantee that the generator will use it correctly.

The project therefore separates:

```text
Retrieval
    ↓
Evidence coverage
    ↓
Evidence composition
    ↓
Generation
    ↓
Verification
```

Each stage can fail independently and can therefore be inspected independently.

---

## Failure Modes Explored During Development

Several RAG failure modes were encountered while building the project:

- relevant evidence not being retrieved
- broad questions causing query drift
- redundant retrieval results
- planner-generated duplicate searches
- malformed search instructions entering retrieval
- structural document content appearing semantically relevant
- context-window pressure
- evidence being retrieved but excluded from generation
- incorrect citation-to-claim alignment
- whole-answer verification producing contradictory rewrites
- claim-level verification becoming overly strict
- final synthesis strengthening claims beyond the evidence

These failures motivated the current separation between retrieval, planning, evidence composition, generation, and verification.

---

## Current Limitations

The project is intentionally experimental and still has several limitations:

- Evidence attribution is performed by an LLM and is therefore probabilistic.
- FAISS relevance thresholds are corpus and embedding-model specific.
- Structural content such as indexes and bibliographies can still enter the candidate pool.
- Broad questions can require several LLM calls and have higher latency.
- The current application handles one active uploaded document at a time.
- Local inference speed depends heavily on available hardware.
- Verification reduces unsupported claims but cannot guarantee factual correctness.

---

## Future Work

Planned experiments include:

- Tool calling
- Agent action/observation loops
- Exposing RAG retrieval as an agent tool
- Deterministic calculator tool
- Document metadata tool
- Multi-document RAG
- Automated RAG evaluation
- Retrieval and verification metrics
- GraphRAG experiments
- Local model experimentation
- LoRA / QLoRA fine-tuning
- vLLM-based model serving

The next major architectural step is turning the existing RAG pipeline into one capability available to a tool-using agent.

---

## Learning Goal

This project was built as a hands-on exploration of production-oriented LLM application architecture.

The goal was not simply to build a PDF chatbot, but to understand where RAG systems fail and how deterministic software components can be used around probabilistic LLM behavior.

The project evolved through multiple iterations:

```text
Basic vector search
        ↓
Persistent FAISS
        ↓
Local LLM generation
        ↓
Agentic follow-up retrieval
        ↓
Evidence requirements
        ↓
Coverage tracking
        ↓
Context budgeting
        ↓
Evidence attribution
        ↓
Citation-scoped verification
        ↓
Arbitrary PDF ingestion
        ↓
Streamlit application
```

---

## License

This project is intended for educational and experimental use.