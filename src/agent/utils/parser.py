import json
import re

from .errors import (
    AgentError,
    ErrorType
)


def parse_json_response(content: str) -> dict:

    content = content.strip()

    if content.startswith("```"):

        content = re.sub(
            r"```json\s*",
            "",
            content
        )

        content = re.sub(
            r"```\s*$",
            "",
            content
        )

    else:

        match = re.search(
            r"\{.*\}",
            content,
            re.S
        )

        if match:
            content = match.group()


    try:

        return json.loads(content)

    except json.JSONDecodeError as e:

        raise AgentError(
            ErrorType.PARSE_ERROR,
            f"JSON解析失败: {e}"
        )