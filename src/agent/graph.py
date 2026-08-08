import os
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


from langgraph.prebuilt import ToolNode, tools_condition
from .tools.github_tool import github_analyzer



from .state import State
from .llm import llm, llm_with_tools
from .nodes.intent import intent_classifier_node

load_dotenv()


confidence_threshold = float(
    os.getenv(
        "INTENT_CONFIDENCE_THRESHOLD",
        "0.8"
    )
)


def call_llm(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"LLM调用失败:{e}"
    
    

def route_task(state: State) -> str:

    task_type = state["task_type"]
    confidence = state["confidence"]

    if confidence < confidence_threshold:
        return "fallback_response"

    if task_type == "resume":
        return "resume_response"

    elif task_type == "project":
        return "project_agent"

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


def project_agent(state: State) -> dict:

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    update = {
        "messages":[response]
    }

    if response.content:
        update["final_answer"] = response.content

    return update


def learning_response(state: State) -> dict:
    analysis_result = state["analysis_result"]
    final_answer = generate_learning_advice(analysis_result)
    message = AIMessage(content=final_answer)


    return {
        "final_answer": final_answer,
        "messages": [message]
    }

def fallback_response(state: State) -> dict:
    user_input = state["user_input"]

    answer = f"""
我暂时无法准确判断您关于：

"{user_input}"

的具体需求。

请进一步说明您的目标：

1. Agent学习路线
2. Agent项目开发
3. 求职和面试准备
"""
    message = AIMessage(content=answer)

    return {
        "final_answer": answer,
        "messages": [message]
    }


tools = [
    github_analyzer
]


tool_node = ToolNode(tools)


workflow = StateGraph(State)
workflow.add_node("intent_classifier",intent_classifier_node)
workflow.add_node("resume_response", resume_response)
workflow.add_node("project_agent",project_agent)
workflow.add_node("learning_response", learning_response)
workflow.add_node("fallback_response",fallback_response)
workflow.add_node("tools",tool_node)

workflow.add_edge(START,"intent_classifier")
workflow.add_conditional_edges("intent_classifier",route_task)
workflow.add_edge("resume_response", END)
workflow.add_edge("learning_response", END)
workflow.add_edge("fallback_response",END)
workflow.add_conditional_edges("project_agent",tools_condition)
workflow.add_edge("tools","project_agent")


graph = workflow.compile()
