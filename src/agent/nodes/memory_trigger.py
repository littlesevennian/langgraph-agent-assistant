from ..state import State
from ..llm import llm
from pydantic import BaseModel
from typing import Optional
from enum import Enum
from pydantic import BaseModel


class MemoryType(str, Enum):

    BASIC_INFO = "basic_info"

    CAREER = "career"

    SKILL = "skill"

    PROJECT = "project"

    LEARNING_INTEREST = "learning_interest"

    NONE = "none"


class MemoryDecision(BaseModel):

    need_memory_update: bool

    memory_types: list[MemoryType]

    confidence: float

    
def memory_trigger_node(state: State):

    user_input = state["user_input"]


    prompt = f"""

判断用户输入是否包含需要长期保存的信息。

长期Memory只保存：

1. basic_info
   用户基本信息：
   姓名、年龄、所在地等


2. career
   用户职业、岗位、工作经历


3. skill
   用户已经掌握的技能


4. project
   用户参与过的项目经历


5. learning_interest
   用户正在学习或者未来想学习的方向


以下情况不要保存：

- 普通知识问题
- 临时调试问题
- 一次性需求


注意：

一个输入可能包含多个类别。


例如：

输入：
"我叫张三，是嵌入式工程师，做过RK2206项目"

输出：

basic_info
career
project

必须返回对应memory_types。

如果need_memory_update为true，
memory_types不能为空。

如果不需要保存，
memory_types返回["none"]。


用户输入:

{user_input}

"""


    structured_llm = llm.with_structured_output(
        MemoryDecision
    )


    result = structured_llm.invoke(prompt)


    return {
        "memory_decision": result.model_dump()
    }