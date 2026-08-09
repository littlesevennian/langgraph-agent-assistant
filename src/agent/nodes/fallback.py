from ..state import State
from ..llm import llm

from langchain_core.messages import AIMessage

def fallback_response(state: State) -> dict:
    user_input = state["user_input"]

    prompt = f"""
我暂时无法准确判断您关于：

"{user_input}"

的具体需求。

请进一步说明您的目标：

1. Agent学习路线
2. Agent项目开发
3. 求职和面试准备
"""

    response = llm.invoke(prompt)
    
    message = AIMessage(content=response.content)

    return {
        "final_answer": response.content,
        "messages": [message]
    }
