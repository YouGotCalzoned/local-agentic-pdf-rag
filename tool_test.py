from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from tools.calculator import calculator


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

tools = [calculator]

llm_with_tools = llm.bind_tools(tools)


question = (
    "A company's revenue increased from 4250 million "
    "to 5600 million. What percentage increase is that?"
)

messages = [
    HumanMessage(content=question)
]


# ---------------------------------------------------------
# STEP 1: Ask the LLM what it wants to do
# ---------------------------------------------------------

response = llm_with_tools.invoke(messages)

messages.append(response)

print("\nLLM RESPONSE:")
print(response.content)

print("\nTOOL CALLS:")
print(response.tool_calls)


# ---------------------------------------------------------
# STEP 2: Execute requested tools
# ---------------------------------------------------------

for tool_call in response.tool_calls:

    if tool_call["name"] == "calculator":

        print("\nEXECUTING TOOL:")
        print(tool_call)

        tool_result = calculator.invoke(tool_call["args"])

        print("\nTOOL RESULT:")
        print(tool_result)

        # Give the result back to the LLM.
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )
        )


# ---------------------------------------------------------
# STEP 3: Give the observation back to the LLM
# ---------------------------------------------------------

final_response = llm_with_tools.invoke(messages)

print("\nFINAL RESPONSE:")
print(final_response.content)