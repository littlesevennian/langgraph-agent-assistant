import time
from .logger import setup_logger

from .errors import (
    AgentError,
    ErrorType
)


logger = setup_logger(
    "agent.llm_client"
)


def invoke_llm(
    runnable,
    input_data,
    max_retries=3,
    retry_delay=1.0,
    caller: str = "unknown"
):

    last_error = None


    for attempt in range(
        max_retries + 1
    ):

        try:
            logger.info(
                "LLM invocation attempt=%s caller=%s",
                attempt + 1,
                caller
            )
            return runnable.invoke(
                input_data
            )


        except Exception as e:

            last_error = e

            retryable = is_retryable_error(e)

            logger.warning(
                "LLM invocation failed "
                "caller=%s attempt=%s retryable=%s error=%s",
                caller,
                attempt + 1,
                retryable,
                e
            )

            # 不可重试异常，立即停止
            if not retryable:
                break

            # 已达到最大重试次数
            if attempt >= max_retries:
                break

            time.sleep(retry_delay)

    logger.error(
        "LLM invocation failed permanently "
        "caller=%s error=%s",
        caller,
        last_error
    )
    
    raise AgentError(
        ErrorType.LLM_ERROR,
        f"LLM调用失败: {last_error}"
    )


def is_retryable_error(error: Exception) -> bool:

    error_message = str(error).lower()

    retryable_keywords = [
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "rate limit",
        "429",
        "503"
    ]

    non_retryable_keywords = [
        "unauthorized",
        "invalid api key",
        "authentication",
        "401",
        "invalid request",
        "400"
    ]

    for keyword in non_retryable_keywords:
        if keyword in error_message:
            return False

    for keyword in retryable_keywords:
        if keyword in error_message:
            return True

    return False