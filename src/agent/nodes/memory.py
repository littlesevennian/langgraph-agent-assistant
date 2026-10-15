from ..state import State
from ..llm import llm
from ..utils.llm_client import invoke_llm


SUMMARY_THRESHOLD = 6


def memory_check_node(state: State):

    messages = state["messages"]

    if len(messages) <= SUMMARY_THRESHOLD:
        return {}


    prompt = f"""
请总结以下对话。

保留：
1. 用户身份
2. 用户目标
3. 用户技术背景
4. 重要限制条件

对话:
{messages}
"""


    response = invoke_llm(
        runnable=llm,
        input_data=prompt,
        caller="memory_agent"
    )


    return {
        "conversation_summary": response.content
    }