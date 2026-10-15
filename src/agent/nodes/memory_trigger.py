from ..state import State
from ..llm import llm
from pydantic import ValidationError
from ..schemas.memory import (
    MemoryDecision,
    MemoryType
)
from ..utils.llm_client import invoke_llm
from ..utils.logger import setup_logger
from ..utils.errors import AgentError,ErrorType
from ..utils.parser import parse_json_response
from ..prompts.memory import MEMORY_TRIGGER_PROMPT



logger = setup_logger(
    "agent.memory_trigger"
)

    
def memory_trigger_node(state: State):

    user_input = state["user_input"]


    prompt = MEMORY_TRIGGER_PROMPT.format(
        user_input=user_input
    )


    memory_llm = llm


    try:
    
        response  = invoke_llm(
            runnable=memory_llm,
            input_data=prompt,
            caller="memory_trigger"
        )

        data=parse_json_response(response.content)

        try:

            memory_decision = MemoryDecision(
                **data
            )

        except ValidationError as e:

            raise AgentError(
            ErrorType.VALIDATION_ERROR,
            f"Memory decision校验失败: {e}"
            )

    except AgentError as e:
        logger.warning(
            "Memory trigger degraded "
            "error=%s",
            e
        )

        return {
            "memory_decision": {
            "need_memory_update": False,
            "memory_types": ["none"],
            "confidence": 0.0
            }
        }

  
    return {
        "memory_decision":
            memory_decision.model_dump(
                mode="json"
            )
    }