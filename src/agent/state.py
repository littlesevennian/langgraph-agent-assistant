from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):

    messages: list

    user_input: str

    task_type: str

    confidence: float

    analysis: str

    final_answer: str


    # Memory Trigger输出
    memory_decision: dict


    # Long-term Memory
    user_profile: dict