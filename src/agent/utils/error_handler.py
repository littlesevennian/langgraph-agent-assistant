from .errors import ErrorType


def get_user_error_message(
    error_type: ErrorType
) -> str:

    mapping = {

        ErrorType.LLM_ERROR:
            "当前AI服务暂时不可用，请稍后重试。",

        ErrorType.TOOL_ERROR:
            "外部服务调用失败，请检查输入信息后重试。",

        ErrorType.PARSE_ERROR:
            "系统处理返回数据时出现异常，请稍后重试。",

        ErrorType.MEMORY_ERROR:
            "用户信息读取失败，但不会影响当前对话，请稍后重试。",

    }


    return mapping.get(
        error_type,
        "系统暂时无法完成请求，请稍后重试。"
    )