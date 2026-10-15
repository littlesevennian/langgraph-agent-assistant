from ..state import State
from ..llm import llm
from ..utils.llm_client import invoke_llm
from ..memory.profile_store import load_profile, save_profile
from ..utils.errors import (AgentError,ErrorType)
from pydantic import ValidationError
from ..utils.logger import setup_logger
from ..utils.parser import parse_json_response
from ..schemas.profile import (
    BasicInfoUpdate,
    CurrentStatusUpdate,
    ProfileUpdate
)
from ..prompts.profile import PROFILE_EXTRACTOR_PROMPT
logger = setup_logger(
    "agent.profile_extractor"
)


def profile_extractor_node(state: State):

    user_input = state["user_input"]

    memory_decision = state.get(
        "memory_decision",
        {}
    )

    memory_types = memory_decision.get(
        "memory_types",
        []
    )

    prompt = PROFILE_EXTRACTOR_PROMPT.format(
        user_input = user_input,
        memory_types = memory_types
    )

    try:

        response = invoke_llm(
            runnable=llm,
            input_data=prompt,
            caller="profile_extractor"
        )

        print("========== MEMORY TYPES ==========")
        print(memory_types)

        print("========== EXTRACT RESULT ==========")
        print(response.content)

        data=parse_json_response(
            response.content
        )
 
        try:
            profile_update = ProfileUpdate(
            **data
            )

        except ValidationError as e:
            raise AgentError(
                ErrorType.VALIDATION_ERROR,
                f"Profile输出校验失败: {e}"
            )

        update_data = profile_update.model_dump(
            exclude_unset=True
        )

        old_profile = load_profile()

        merged_profile = merge_profile(
            old_profile,
            update_data
        )

        try:
            save_profile(
            merged_profile
        )

        except Exception as e:
            logger.warning(
            f"Profile save failed error={e}"
            )
            return {}

        return {
             "profile_update": profile_update
        }

    except AgentError as e:

        logger.warning(
            "Profile extractor degraded error=%s",
            e
        )

        return {}


def merge_profile(old, new):

    if not old:
        return new


    for key, value in new.items():

        # 新字段
        if key not in old:
            old[key] = value
            continue


        old_value = old[key]


        # 字典合并
        if isinstance(value, dict):

            if isinstance(old_value, dict):

                for k, v in value.items():

                    if v is not None:
                        old_value[k] = v

            else:
                old[key] = value

           


        # 列表合并
        elif isinstance(value, list):

            if not isinstance(old_value, list):
                old[key] = value

            else:

                for item in value:

                    if item not in old_value:
                        old_value.append(item)


        # 普通字段覆盖
        else:

            old[key] = value


    return old


