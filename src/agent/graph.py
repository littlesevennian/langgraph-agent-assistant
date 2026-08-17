import os


from dotenv import load_dotenv

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


from .tools.github_tool import github_analyzer

from .state import State

from .nodes.intent import intent_classifier_node

from .nodes.resume import resume_response
from .nodes.learning import learning_response
from .nodes.fallback import fallback_response
from .nodes.project import project_agent

from .nodes.memory import memory_check_node
from .nodes.memory_query import memory_query_response
from .nodes.memory_trigger import memory_trigger_node
from .nodes.profile_extractor import profile_extractor_node

from langgraph.checkpoint.memory import MemorySaver


load_dotenv()


confidence_threshold = float(
    os.getenv(
        "INTENT_CONFIDENCE_THRESHOLD",
        "0.8"
    )
)


def route_memory(state):

    decision = state.get(
        "memory_decision",
        {}
    )


    if decision.get(
        "need_memory_update",
        False
    ):
        return "profile_extractor"


    return "intent_classifier"


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

    elif task_type == "memory_query":
        return "memory_query"

    else:
        return "fallback_response"
    
    

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
workflow.add_node("memory",memory_check_node)
workflow.add_node("memory_query",memory_query_response)
workflow.add_node("memory_trigger",memory_trigger_node)
workflow.add_node("profile_extractor",profile_extractor_node)

workflow.add_edge(START,"memory_trigger")
workflow.add_conditional_edges(

    "memory_trigger",

    route_memory,

    {

        "profile_extractor":
        "profile_extractor",


        "intent_classifier":
        "intent_classifier"

    }

)
workflow.add_edge( "profile_extractor","intent_classifier")
workflow.add_edge("memory","intent_classifier")
workflow.add_conditional_edges(

    "intent_classifier",

    route_task,

    {

        "project_agent":
        "project_agent",

        "resume_response":
        "resume_response",

        "learning_response":
        "learning_response",

        "memory_query":
        "memory_query",

        "fallback_response":
        "fallback_response"

    }
)
workflow.add_edge("resume_response", END)
workflow.add_edge("learning_response", END)
workflow.add_edge( "memory_query",END)
workflow.add_edge("fallback_response",END)
workflow.add_conditional_edges("project_agent",tools_condition)
workflow.add_edge("tools","project_agent")



memory = MemorySaver()

graph = workflow.compile(
    checkpointer=memory
)
