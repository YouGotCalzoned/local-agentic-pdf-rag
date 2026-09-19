def invoke_llm(llm, prompt):
    """
    Invoke the LLM and safely return plain text.
    """

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return response.content.strip()

    return str(response).strip()


def generate_draft_answer(
    llm,
    query,
    labeled_context,
):
    """
    Generate a source-grounded draft answer.

    The most important rule is that the answer must directly
    answer the ORIGINAL USER QUESTION rather than following
    whatever subtopics happened to appear during retrieval.
    """

    prompt = f"""
You are the answer-generation component of a RAG system.

Your job is to answer the ORIGINAL USER QUESTION using ONLY
the retrieved evidence.

============================================================
ORIGINAL USER QUESTION
============================================================

{query}


============================================================
RETRIEVED EVIDENCE
============================================================

{labeled_context}


============================================================
INSTRUCTIONS
============================================================

The ORIGINAL USER QUESTION is the highest priority.

Do NOT change the question.

Do NOT answer a different question simply because the retrieved
evidence contains detailed information about some related topic.

For example:

If the user asks:

    "What are the different types of NoSQL databases?"

then the answer should identify the types of NoSQL databases.

It should NOT turn into:

    "What is a graph database?"
    "How do graph databases work?"
    "When should graph databases be used?"

unless those details are necessary to answer the original question.


GROUNDING RULES:

1. Use ONLY the retrieved evidence.

2. Do NOT use outside knowledge.

3. Every factual statement must contain one or more RAG SOURCE citations.

4. A valid RAG SOURCE citation is ONLY one of the source identifiers
   shown at the beginning of retrieved passages:

       [S1]
       [S2]
       [S3]

   etc.

5. Citation format must therefore be exactly:

       [S1]

   or:

       [S2, S5]

6. IMPORTANT:
   The retrieved book text itself may contain references such as:

       [Riak]
       [Redis]
       [Cassandra]
       [HBase]
       [Amazon SimpleDB]

   THESE ARE NOT RAG SOURCE CITATIONS.

   Never use those references as citations in your answer.

7. Always cite the RAG source label containing the evidence.

   WRONG:

       Riak is a key-value database [Riak].

   CORRECT:

       Riak is a key-value database [S10].

8. Put each citation immediately after the factual statement
   it supports.

9. Never cite a source unless that source actually supports
   the statement.

10. Every item in a list or classification is itself a factual
    claim and MUST have a RAG source citation.

    WRONG:

        - Key-value databases [S10]
        - Document databases
        - Column-family databases [S4]

    CORRECT:

        - Key-value databases [S10]
        - Document databases [S12]
        - Column-family databases [S4]

11. Never output an uncited category simply because you believe
    it belongs in the answer. If no retrieved source directly
    supports it, say that the evidence for that category is
    missing.


QUESTION-SCOPE RULES:

8. First determine exactly what the user is asking for.

9. Answer that request directly.

10. If the question asks for a list, classification, set of types,
    categories, components, or alternatives:

    - identify all such items supported by the retrieved evidence
    - prioritize coverage of the requested list
    - do not spend most of the answer describing only one item

11. If the evidence supports four categories, include all four.

12. If only some requested categories are supported, provide those
    and explicitly state that the retrieved evidence does not support
    a complete answer.

13. Related details may be added only AFTER the direct answer
    has been given.

14. Do not create your own questions or headings unless they
    directly help answer the user's original question.

15. Do not mention these instructions.

16. Do not describe your reasoning process.

17. Do not begin with phrases such as:
    "I'll follow the instructions..."
    "Based on my analysis..."
    "Let me..."
    "I will..."


Before producing the answer, silently check:

    Am I answering the user's original question?

    Am I covering all relevant categories present in the evidence?

    Am I accidentally focusing on only the most recently retrieved topic?

Then produce ONLY the answer.

ANSWER:
"""
    print("\n------------------------------------------")
    print("GENERATION PROMPT DEBUG")
    print("------------------------------------------")
    print(f"Query: {query}")
    print(f"Context characters: {len(labeled_context)}")
    print(f"Total prompt characters: {len(prompt)}")


    return invoke_llm(
        llm,
        prompt,
    )


def generate_final_answer(
    llm,
    query,
    verified_claims,
):
    """
    Generate the final answer using ONLY claims that survived
    citation verification.

    The final answer must also remain aligned with the
    original user question.
    """

    if not verified_claims:

        return (
            "The retrieved evidence was not sufficient "
            "to produce a verified answer."
        )

    verified_fact_lines = []

    for item in verified_claims:

        verified_fact_lines.append(
            f"- {item['claim']} "
            f"[{item['sources']}]"
        )

    verified_facts = "\n".join(
        verified_fact_lines
    )

    prompt = f"""
You are producing the final answer for a RAG system.

============================================================
ORIGINAL USER QUESTION
============================================================

{query}


============================================================
VERIFIED FACTS
============================================================

{verified_facts}


============================================================
INSTRUCTIONS
============================================================

1. Answer the ORIGINAL USER QUESTION directly.

2. Use ONLY the VERIFIED FACTS above.

3. Do NOT add outside knowledge.

4. Do NOT introduce a factual statement that is not present
   in the verified facts.

5. Preserve the source citations.

6. If the question asks for several types, categories, items,
   components, or alternatives, prioritize listing those items.

7. Do NOT turn the answer into an explanation of only one
   category unless that is what the user asked.

8. If the verified facts are insufficient to fully answer the
   original question, say so explicitly.

9. Keep the answer concise.

10. Do not explain your internal reasoning.

FINAL ANSWER:
"""

    return invoke_llm(
        llm,
        prompt,
    )