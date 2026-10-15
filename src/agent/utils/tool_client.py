import time

from .errors import (
    AgentError,
    ErrorType
)
from .logger import setup_logger


logger = setup_logger(
    "agent.tool_client"
)


def is_retryable_tool_error(
    error: Exception
) -> bool:

    error_message = str(error).lower()

    non_retryable_keywords = [
        "404",
        "not found",
        "repository not found",
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "invalid argument",
        "bad request"
    ]

    retryable_keywords = [
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504"
    ]

    for keyword in non_retryable_keywords:
        if keyword in error_message:
            return False

    for keyword in retryable_keywords:
        if keyword in error_message:
            return True

    return False


def invoke_tool(
    tool,
    input_data,
    max_retries=2,
    retry_delay=1.0
):

    last_error = None

    # 获取Tool名称
    # LangChain Tool通常有name属性
    # Fake Tool没有name时，就使用类名
    tool_name = getattr(
        tool,
        "name",
        tool.__class__.__name__
    )

    for attempt in range(
        max_retries + 1
    ):

        try:
            logger.info(
                "Tool invocation attempt=%s tool=%s",
                attempt + 1,
                tool_name
            )

            return tool.invoke(
                input_data
            )

        except Exception as e:

            last_error = e

            retryable = is_retryable_tool_error(
                e
            )

            logger.warning(
                "Tool invocation failed "
                "tool=%s attempt=%s "
                "retryable=%s error=%s",
                tool_name,
                attempt + 1,
                retryable,
                e
            )

            if not retryable:
                break

            if attempt >= max_retries:
                break

            time.sleep(
                retry_delay
            )

    logger.error(
        "Tool invocation failed permanently "
        "tool=%s error=%s",
        tool_name,
        last_error
    )

    raise AgentError(
        ErrorType.TOOL_ERROR,
        f"Tool调用失败: {last_error}"
    )