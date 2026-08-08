from ..state import State
from ..llm import llm
from pydantic import BaseModel
from langchain_core.messages import HumanMessage


class IntentResult(BaseModel):
    task_type: str
    confidence: float
    reason: str


structured_llm = llm.with_structured_output(IntentResult)


def intent_classifier_node(state: State) -> dict:

    user_input = state["user_input"]

    prompt = f"""
你是一个任务分类Agent。

根据用户输入判断任务类型。

只能从以下三个类型选择：

resume:
求职、简历优化、面试准备。

project:
项目开发、GitHub项目复刻、项目分析。

learning:
Agent学习、技术路线规划、概念学习。

返回JSON格式。


用户输入：

{user_input}


输出要求：

1. task_type必须是三个选项之一。
2. confidence必须在0到1之间。
3. reason说明判断依据。
4. 不输出额外字段。
"""
    
    result = structured_llm.invoke(prompt)

    return {
        "task_type": result.task_type,
        "confidence": result.confidence,
        "analysis_result": result.reason,
        "messages": [HumanMessage(content=user_input)]
    }

