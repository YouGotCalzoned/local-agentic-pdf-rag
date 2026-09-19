# ============================================================
# MODEL CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

OLLAMA_MODEL_NAME = "llama3.1:8b"

FAISS_INDEX_PATH = "faiss_index"


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

# Number of nearest chunks inspected during the initial
# similarity search.
INITIAL_TOP_K = 3

# Number of chunks returned for each agent-generated
# follow-up query.
FOLLOW_UP_TOP_K = 3

# Maximum number of agentic retrieval rounds.
#
# This prevents the planner from entering an uncontrolled
# search loop.
MAX_RETRIEVAL_ROUNDS = 3

# Maximum number of follow-up searches the planner may
# generate in one retrieval round.
MAX_FOLLOW_UP_QUERIES = 3


# ============================================================
# RELEVANCE GATING
# ============================================================

# FAISS here behaves like a distance:
#
#     lower = more similar / better
#     higher = less similar / worse
#
# This threshold was chosen experimentally for the current
# NoSQL Distilled corpus.
#
# IMPORTANT:
# This is NOT a universal threshold.
RELEVANCE_DISTANCE_THRESHOLD = 1.20


# ============================================================
# INITIAL RETRIEVAL MMR
# ============================================================

# Number of candidates FAISS may inspect before MMR chooses
# the smaller diverse result set.
MMR_FETCH_K = 15

# Initial retrieval MMR balance.
#
# 1.0 = pure relevance
# 0.0 = pure diversity
MMR_LAMBDA = 0.5


# ============================================================
# GENERATION CONTEXT BUDGET
# ============================================================

# Maximum number of chunks eventually shown to the generator.
MAX_GENERATION_CHUNKS = 8

# Approximate raw evidence-character budget.
MAX_GENERATION_CONTEXT_CHARS = 10000


# ============================================================
# GENERATION-STAGE MMR
# ============================================================

# We are currently experimenting with a second MMR stage
# after agentic retrieval.
#
# This stage operates on ALL evidence discovered by the agent
# and chooses the final evidence shown to the generator.
GENERATION_MMR_LAMBDA = 0.7


# ============================================================
# COVERAGE TRACKING
# ============================================================

# Maximum number of evidence requirements the planner is
# allowed to derive from a question.
#
# We deliberately keep this small.
#
# Example:
#
# Question:
# "Explain how replication, consistency, and availability
# are related."
#
# Possible requirements:
#
# R1 replication basics
# R2 replication -> consistency
# R3 replication -> availability
# R4 consistency/availability tradeoff
# R5 CAP/network partitions
#
MAX_EVIDENCE_REQUIREMENTS = 6