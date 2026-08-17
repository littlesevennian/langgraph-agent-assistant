from ..state import State
from ..llm import llm
from pydantic import BaseModel


class IntentResult(BaseModel):
    task_type: str
    confidence: float
    reason: str


structured_llm = llm.with_structured_output(IntentResult)


def intent_classifier_node(state: State) -> dict:
    user_input = state["user_input"]
    messages = state["messages"]

    prompt = f"""
你是一个任务分类Agent。

请结合历史对话和当前用户输入，判断当前用户需求。

任务类型只能从以下类型选择：

resume:
求职、简历优化、面试准备。

project:
项目开发、GitHub项目复刻、项目分析。

learning:
Agent学习、技术路线规划、概念学习。

memory_query:
询问历史对话中的个人信息、背景、之前提到的信息。
例如：
- 我的职业是什么？
- 我之前告诉你什么？
- 我的项目是什么？
- 你还记得我的背景吗？


历史对话：
{messages}

当前用户输入：
{user_input}


输出要求：

1. task_type必须是resume、project、learning之一。
2. confidence必须在0到1之间。
3. reason说明判断依据。
4. 判断当前输入时，应结合历史对话理解省略的信息。
"""

    result = structured_llm.invoke(prompt)

    return {
        "task_type": result.task_type,
        "confidence": result.confidence,
        "analysis": result.reason,
    }

