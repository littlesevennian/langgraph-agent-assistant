import json
import os

from ..utils.errors import AgentError, ErrorType


PROFILE_PATH = "user_profile.json"


def load_profile():

    if not os.path.exists(PROFILE_PATH):
        return {}

    try:

        with open(
            PROFILE_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except json.JSONDecodeError as e:

        raise AgentError(
            ErrorType.MEMORY_ERROR,
            f"Profile JSON解析失败: {e}"
        ) from e


    except OSError as e:

        raise AgentError(
            ErrorType.MEMORY_ERROR,
            f"Profile读取失败: {e}"
        ) from e


def save_profile(profile):

    try:

        with open(
            PROFILE_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                profile,
                f,
                ensure_ascii=False,
                indent=4
            )


    except OSError as e:

        raise AgentError(
            ErrorType.MEMORY_ERROR,
            f"Profile保存失败: {e}"
        ) from e