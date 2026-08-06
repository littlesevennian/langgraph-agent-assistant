from langchain_core.tools import tool


@tool
def github_analyzer(repo_name: str) -> dict:
    """
    分析GitHub仓库基础信息。

    当用户需要分析GitHub项目结构、
    技术栈或仓库信息时调用。

    参数:
        repo_name: GitHub仓库名称

    返回:
        仓库基础信息
    """

    return {
        "name": repo_name,
        "language": "Python",
        "files": 50,
        "description": "LangGraph Agent project"
    }