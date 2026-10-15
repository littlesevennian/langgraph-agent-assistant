from ..state import State
from ..llm import llm_with_tools
from ..tools.github_tool import github_analyzer
from ..utils.llm_client import invoke_llm
from langchain_core.messages import ToolMessage
from ..utils.tool_client import invoke_tool
from ..utils.logger import setup_logger
from ..utils.errors import AgentError,ErrorType


logger = setup_logger(
    "agent.project_agent"
)


def project_agent(state: State) -> dict:

    messages = state["messages"]

    response = invoke_llm(
        runnable=llm_with_tools,
        input_data=messages,
        caller="project_agent.tool_decision"
    )


    # 判断LLM是否请求调用工具
    if response.tool_calls:

        tool_messages = []


        for tool_call in response.tool_calls:

            if tool_call["name"] == "github_analyzer":

                try:
                    tool_result = invoke_tool(
                        tool=github_analyzer,
                        input_data=tool_call["args"]
                    )
                    
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )

                except AgentError as e:
                    logger.warning(
                        "Project agent tool degraded "
                        "tool=%s error=%s",
                        tool_call["name"],
                        e
                    )

                    tool_message = ToolMessage(
                        content=(
                            "GitHub tool execution failed. "
                            "Repository data is unavailable. "
                            "Do not fabricate repository-specific facts. "
                            "Explain the limitation clearly and only provide "
                            "general analysis that does not depend on repository data."
                        ),
                        tool_call_id=tool_call["id"]
                    )

                tool_messages.append(
                    tool_message
                )


        updated_messages = messages + [
            response,
            *tool_messages
        ]


        final_response = invoke_llm(
            runnable=llm_with_tools,
            input_data=updated_messages,
            caller="project_agent.tool_summary"
        )


        return {
            "messages":[
                response,
                *tool_messages,
                final_response
            ],
            "final_answer": final_response.content
        }


    return {
        "messages":[response],
        "final_answer":response.content
    }