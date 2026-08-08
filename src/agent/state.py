from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):

    user_input: str

    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]

    task_type: str

    confidence: float

    analysis_result: str

    final_answer: str