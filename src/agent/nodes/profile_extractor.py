from ..state import State
from ..llm import llm
from ..memory.profile_store import load_profile, save_profile

import json
import re


def clean_json_response(content):

    content = content.strip()

    # 去除 markdown json代码块
    if content.startswith("```"):

        content = re.sub(
            r"```json\s*",
            "",
            content
        )

        content = re.sub(
            r"```\s*$",
            "",
            content
        )

    return content.strip()


def profile_extractor_node(state: State):

    user_input = state["user_input"]

    memory_decision = state.get(
    "memory_decision",
    {}
    )

    memory_types = memory_decision.get(
    "memory_types",
    []
    )

    prompt = f"""

你是一个用户信息提取助手。

请根据指定的信息类型，
从用户输入中提取长期记忆。


允许提取的信息类型：

{memory_types}


类型说明：

basic_info:
姓名、年龄、所在地等基础信息

career:
职业、岗位、工作经历

skill:
已经掌握的技能

project:
项目经历

learning_interest:
正在学习方向


要求：

1. 只提取上述指定类型的信息
2. 不要提取其他信息
3. 如果没有对应信息，返回空


用户输入：

{user_input}


请严格按照以下JSON格式返回：

{{
    "basic_info": {{}},

    "current_status": {{}},

    "skills": [],

    "projects": [],

    "learning_interest": []
}}


规则：

career信息必须放在current_status中。

例如：

输入：
我是嵌入式工程师

输出：

{{
    "current_status":{{
        "career":"嵌入式工程师"
    }}
}}

"""


    response = llm.invoke(prompt)

    print("========== MEMORY TYPES ==========")
    print(memory_types)

    print("========== EXTRACT RESULT ==========")
    print(response.content)

    clean_content = clean_json_response(
    response.content
    )


    extracted_profile = json.loads(
    clean_content
    )


    old_profile = load_profile()


    new_profile = merge_profile(
        old_profile,
        extracted_profile
    )


    save_profile(new_profile)


    return {
        "user_profile": new_profile
    }


def merge_profile(old, new):

    if not old:
        return new


    for key, value in new.items():

        # 新字段
        if key not in old:
            old[key] = value
            continue


        old_value = old[key]


        # 字典合并
        if isinstance(value, dict):

            if isinstance(old_value, dict):

                old_value.update(value)

            else:
                old[key] = value


        # 列表合并
        elif isinstance(value, list):

            if not isinstance(old_value, list):
                old[key] = value

            else:

                for item in value:

                    if item not in old_value:
                        old_value.append(item)


        # 普通字段覆盖
        else:

            old[key] = value


    return old


