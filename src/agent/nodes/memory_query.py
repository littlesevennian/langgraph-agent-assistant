from ..llm import llm
from ..memory.profile_store import load_profile
def memory_query_response(state):

    messages = state["messages"]
    profile = load_profile()

    prompt = f"""

你是一个记忆查询助手。


长期用户信息:

{profile}


近期对话:

{messages}


当前问题:

{state["user_input"]}


请优先使用长期用户信息回答。

如果长期信息和历史对话都没有，
明确说明不知道。

"""

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content
    }