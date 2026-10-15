from ..state import State
from ..llm import llm
from langchain_core.messages import AIMessage
from ..utils.llm_client import invoke_llm
from ..prompts.resume import RESUME_PROMPT

def generate_resume_advice(analysis_result: str) -> str:

    prompt = RESUME_PROMPT.format(
        analysis_result = analysis_result
    )

    response = invoke_llm(
        runnable=llm,
        input_data=prompt,
        caller="resume_agent"
    )

    return response.content


def resume_response(state: State) -> dict:
    analysis_result = state["analysis"]
    final_answer = generate_resume_advice(analysis_result)
    message = AIMessage(content=final_answer)


    return {
        "final_answer": final_answer,
        "messages": [message]
    }