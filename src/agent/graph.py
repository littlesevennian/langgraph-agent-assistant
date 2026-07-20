import os
from dotenv import load_dotenv

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


load_dotenv()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    task_type: str
    analysis_result: str
    final_answer: str


def analyze_input(state: State) -> dict:
    user_input = state["user_input"]
    user_input_lower = user_input.lower()
    if "求职" in user_input or "简历" in user_input or "找工作" in user_input:
        task_type = "resume"
        analysis_result = "用户目标是求职准备。"
    elif "github" in user_input_lower or "项目" in user_input_lower or "复刻" in user_input_lower:
        task_type = "project"
        analysis_result = "用户目标是复刻 GitHub 项目。"
    else:
        task_type = "learning"
        analysis_result = "用户目标是 Agent 学习。"
    

    return {
        "task_type": task_type,
        "analysis_result": analysis_result
    }


def call_llm(prompt: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("DASHSCOPE_MODEL", "qwen3-max")
    temperature = float(os.getenv("DASHSCOPE_TEMPERATURE", "0.7"))

    if not api_key:
        return f"当前未配置 API Key，以下为本地兜底结果：{prompt[:100]}..."

    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )
        response = llm.invoke(prompt)

        return response.content
    except Exception as e:
        return f"LLM 调用失败，已返回本地兜底结果。错误信息：{str(e)}"
    
    

def route_task(state: State) -> str:
    task_type = state["task_type"]
    if task_type == "resume":
        return "resume_response"
    elif task_type == "project":
        return "project_response"
    else:
        return "learning_response"
    

def generate_resume_advice(analysis_result: str) -> str:
    prompt = f"""
你是一个 Agent 项目简历优化助手。

当前用户分析结果：
{analysis_result}

请根据这个分析结果，给用户生成一段求职和简历优化建议。

要求：
1. 说明这个 Agent 项目应该如何写成简历项目经历。
2. 给出 3 个简历表达重点。
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

workflow = StateGraph(State)
workflow.add_node("analyze_input", analyze_input)
workflow.add_node("resume_response", resume_response)
workflow.add_node("project_response", project_response)
workflow.add_node("learning_response", learning_response)
workflow.add_edge(START, "analyze_input")
workflow.add_conditional_edges("analyze_input", route_task)
workflow.add_edge("resume_response", END)
workflow.add_edge("project_response", END)
workflow.add_edge("learning_response", END)


graph = workflow.compile()
