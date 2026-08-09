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

from .nodes.resume import resume_response

from .nodes.learning import learning_response

from .nodes.fallback import fallback_response

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
    

def project_agent(state: State) -> dict:

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    update = {
        "messages":[response]
    }

    if response.content:
        update["final_answer"] = response.content

    return update




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
