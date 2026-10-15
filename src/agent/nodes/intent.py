from ..state import State
from ..llm import llm
from ..utils.llm_client import invoke_llm
from ..utils.logger import setup_logger
from ..utils.errors import (AgentError, ErrorType)
from ..utils.parser import parse_json_response
from typing import Literal
from pydantic import ValidationError
from ..schemas.intent import IntentResult
from ..prompts.intent import INTENT_PROMPT

logger = setup_logger(
    "agent.intent_classifier"
)



intent_llm = llm


def classify_intent(user_input: str, messages: list) -> IntentResult:
    
    prompt = INTENT_PROMPT.format(
    messages=messages,
    user_input=user_input
)

    
    result = invoke_llm(
        runnable=intent_llm,
        input_data=prompt,
        caller="intent_classifier"
    )

    try:
        data = parse_json_response(result.content)
    except AgentError as e:
        logger.warning(
            "Intent parse failed raw_content=%r error=%s",
            result.content,
            e
        )
        raise
      
    try:

        intent_result = IntentResult(**data)

    except ValidationError as e:

        raise AgentError(
            ErrorType.VALIDATION_ERROR,
            f"Intent输出校验失败: {e}"
        )from e
    
    return intent_result

    
def intent_classifier_node(state: State) -> dict:
    user_input = state["user_input"]
    messages = state["messages"]

    try:
        intent_result = classify_intent(
            user_input=user_input,
            messages=messages
        )

    except AgentError as e:
        logger.warning(
            "Intent classifier degraded "
            "route=fallback error=%s",
            e
        )

        return {
            "task_type": "fallback",
            "confidence": 0.0,
            "analysis": (
                "Intent classification failed; "
                "routed to fallback safely."
            )
        }

    return {
        "task_type": intent_result.task_type,
        "confidence": intent_result.confidence,
        "analysis": intent_result.reason
    }
