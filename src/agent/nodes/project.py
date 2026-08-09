from ..state import State
from ..llm import llm_with_tools

def project_agent(state: State) -> dict:

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    update = {
        "messages":[response]
    }

    if response.content:
        update["final_answer"] = response.content

    return update