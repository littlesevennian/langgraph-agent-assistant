from ..state import State
from ..llm import llm
from langchain_core.messages import AIMessage
def generate_resume_advice(analysis_result: str) -> str:

    prompt = f"""
你是一个 Agent 项目简历优化助手。

当前用户分析结果：

{analysis_result}


请根据这个分析结果，给用户生成求职和简历优化建议。


要求：

1. 说明这个Agent项目应该如何写成简历项目经历。

2. 给出3个简历表达重点。

3. 语言具体，不要空泛。

4. 不要编造用户没有提供的技术栈。
"""

    response = llm.invoke(prompt)

    return response.content


def resume_response(state: State) -> dict:
    analysis_result = state["analysis"]
    final_answer = generate_resume_advice(analysis_result)
    message = AIMessage(content=final_answer)


    return {
        "final_answer": final_answer,
        "messages": [message]
    }