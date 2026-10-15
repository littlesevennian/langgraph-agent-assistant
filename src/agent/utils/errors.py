from enum import Enum


class ErrorType(Enum):
    """
    Agent系统统一错误类型。

    用于区分不同来源的异常，
    方便：
    1. Retry策略判断
    2. 日志分类
    3. 全局异常响应
    4. 后续监控统计
    """

    # 大模型调用失败
    LLM_ERROR = "llm_error"

    # Tool调用失败
    TOOL_ERROR = "tool_error"

    # 数据解析失败
    PARSE_ERROR = "parse_error"

    # Memory相关失败
    MEMORY_ERROR = "memory_error"

    # 未分类错误
    UNKNOWN_ERROR = "unknown_error"

    # 数据验证失败
    VALIDATION_ERROR = "validation_error"



class AgentError(Exception):

    def __init__(
        self,
        error_type: ErrorType,
        message: str
    ):

        self.error_type = error_type

        self.message = message

        super().__init__(message)


    def __str__(self):

        return (
            f"{self.error_type.value}: "
            f"{self.message}"
        )