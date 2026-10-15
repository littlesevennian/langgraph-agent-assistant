from ..llm import llm
from ..memory.profile_store import load_profile
from ..utils.llm_client import invoke_llm
from ..prompts.memory_query import MEMORY_QUERY_PROMPT


def memory_query_response(state):

    messages = state["messages"]
    profile = load_profile()
    user_input = state["user_input"]

    prompt = MEMORY_QUERY_PROMPT.format(
        profile=profile,
        messages=messages,
        user_input=user_input
    )

    response = invoke_llm(
        runnable=llm,
        input_data=prompt,
        caller="memory_query"
    )

    return {
        "final_answer": response.content
    }