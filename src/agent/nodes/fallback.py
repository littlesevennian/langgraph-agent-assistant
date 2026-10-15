from ..state import State
from ..llm import llm
from ..utils.llm_client import invoke_llm

from langchain_core.messages import AIMessage
from ..prompts.fallback import FALLBACK_PROMPT

def fallback_response(state: State) -> dict:
    user_input = state["user_input"]

    prompt = FALLBACK_PROMPT.format(
        user_input=user_input
    )

    response = invoke_llm(
        runnable=llm,
        input_data=prompt,
        caller="fallback_agent"
    )
    
    message = AIMessage(content=response.content)

    return {
        "final_answer": response.content,
        "messages": [message]
    }
