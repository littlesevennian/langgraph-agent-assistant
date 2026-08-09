from ..state import State
from ..llm import llm

from langchain_core.messages import AIMessage

def generate_learning_advice(analysis_result: str) -> str:
    prompt = f"""
你是一个 Agent 学习路线规划助手。

当前用户分析结果：
{analysis_result}

请根据这个分析结果，给用户生成一段 Agent 学习建议。

要求：
1. 说明用户当前应该补充哪个基础概念。
2. 给出 3 个练习任务。
3. 语言清晰，避免复杂术语。
4. 不要直接跳到高级框架源码。
"""
    
    response = llm.invoke(prompt)

    return response.content



def learning_response(state: State) -> dict:
    analysis_result = state["analysis_result"]
    final_answer = generate_learning_advice(analysis_result)
    message = AIMessage(content=final_answer)


    return {
        "final_answer": final_answer,
        "messages": [message]
    }
