from ..state import State
from ..llm import llm
from ..utils.llm_client import invoke_llm
from langchain_core.messages import AIMessage
from ..utils.errors import AgentError,ErrorType
from ..prompts.learning import LEARNING_PROMPT



def generate_learning_advice(analysis_result: str) -> str:

    prompt = LEARNING_PROMPT.format(
        analysis_result = analysis_result
    )

    response = invoke_llm(
        runnable=llm,
        input_data=prompt,
        caller="learning_agent"
    )

    return response.content



def learning_response(state: State) -> dict:
    analysis_result = state["analysis"]
    final_answer = generate_learning_advice(analysis_result)
    message = AIMessage(content=final_answer)


    return {
        "final_answer": final_answer,
        "messages": [message]
    }
