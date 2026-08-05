import os
from dotenv import load_dotenv

from collections.abc import Sequence
from typing import Annotated, TypedDict
from pydantic import BaseModel

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


load_dotenv()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    task_type: str
    confidence: float
    analysis_result: str
    final_answer: str


class IntentResult(BaseModel):
    task_type: str
    confidence: float
    reason: str


api_key = os.getenv("DASHSCOPE_API_KEY")

base_url = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

model = os.getenv(
    "DASHSCOPE_MODEL",
    "qwen3.7-max"
)

temperature = float(
    os.getenv(
        "DASHSCOPE_TEMPERATURE",
        "0.7"
    )
)

if not api_key:
    raise ValueError("DASHSCOPE_API_KEY未配置")

llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url,
    temperature=temperature
)

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
        "analysis_result": result.reason
    }


def call_llm(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"LLM调用失败:{e}"
    
    

def route_task(state: State) -> str:

    task_type = state["task_type"]
    confidence = state["confidence"]

    if confidence < 0.8:
        return "fallback_response"

    if task_type == "resume":
        return "resume_response"

    elif task_type == "project":
        return "project_response"

    elif task_type == "learning":
        return "learning_response"

    else:
        return "fallback_response"
    

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

    return call_llm(prompt)



def generate_project_advice(analysis_result: str) -> str:
    prompt = f"""
你是一个 Agent 项目训练助手。

当前用户分析结果：
{analysis_result}

请根据这个分析结果，给用户生成一段项目复刻建议。

要求：
1. 说明用户当前应该先做什么。
2. 给出 3 个具体步骤。
3. 语言简洁，适合初学者理解。
4. 不要编造用户没有提供的信息。
"""
    
    return call_llm(prompt)


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
    
    return call_llm(prompt)


def resume_response(state: State) -> dict:
    analysis_result = state["analysis_result"]
    final_answer = generate_resume_advice(analysis_result)
    message = AIMessage(content=final_answer)


    return {
        "final_answer": final_answer,
        "messages": [message]
    }


def project_response(state: State) -> dict:
    analysis_result = state["analysis_result"]
    final_answer = generate_project_advice(analysis_result)
    message = AIMessage(content=final_answer)


    return {
        "final_answer": final_answer,
        "messages": [message]
    }


def learning_response(state: State) -> dict:
    analysis_result = state["analysis_result"]
    final_answer = generate_learning_advice(analysis_result)
    message = AIMessage(content=final_answer)


    return {
        "final_answer": final_answer,
        "messages": [message]
    }

def fallback_response(state: State):

    user_input = state["user_input"]

    answer=f"""
我目前无法准确判断：

"{user_input}"

您的目标。

请进一步说明：

1. 是否希望开发Agent项目？
2. 是否希望学习Agent技术？
3. 是否希望用于求职？
"""


workflow = StateGraph(State)
workflow.add_node("intent_classifier",intent_classifier_node)
workflow.add_node("resume_response", resume_response)
workflow.add_node("project_response", project_response)
workflow.add_node("learning_response", learning_response)
workflow.add_node("fallback_response",fallback_response)
workflow.add_edge(START,"intent_classifier")
workflow.add_conditional_edges("intent_classifier",route_task)
workflow.add_edge("resume_response", END)
workflow.add_edge("project_response", END)
workflow.add_edge("learning_response", END)
workflow.add_edge("fallback_response",END)


graph = workflow.compile()
