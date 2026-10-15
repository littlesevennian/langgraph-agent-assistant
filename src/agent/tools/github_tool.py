
import json
import os
import re
from dotenv import load_dotenv
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from langchain_core.tools import tool

load_dotenv()
def normalize_repo_name(repo_name: str) -> str:
    """支持 owner/repo 或 GitHub 仓库 URL。"""
    value = repo_name.strip()

    if value.startswith(("https://", "http://")):
        parsed = urlparse(value)
        if parsed.hostname not in ("github.com", "www.github.com"):
            raise ValueError("仅支持 GitHub 仓库链接")
        value = parsed.path.strip("/")

    value = value.removesuffix(".git").strip("/")
    parts = value.split("/")

    if len(parts) != 2 or not all(
        re.fullmatch(r"[A-Za-z0-9_.-]+", part)
        for part in parts
    ):
        raise ValueError("仓库格式应为 owner/repo")

    return "/".join(parts)


def github_get(path: str):
    """访问 GitHub REST API。"""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "langgraph-agent-assistant",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        f"https://api.github.com{path}",
        headers=headers
    )

    try:
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    except HTTPError as error:
        raise RuntimeError(
            f"GitHub API HTTP {error.code}"
        ) from error

    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            f"GitHub API 连接失败: {error}"
        ) from error


@tool
def github_analyzer(repo_name: str) -> dict:
    """
    获取 GitHub 公开仓库的真实基础信息。

    当用户要求分析特定 GitHub 仓库时使用。

    repo_name:
        owner/repo 或 GitHub 仓库链接。
    """
    full_name = normalize_repo_name(repo_name)

    encoded_name = "/".join(
        quote(part, safe="")
        for part in full_name.split("/")
    )

    repo = github_get(f"/repos/{encoded_name}")

    root_contents = github_get(
        f"/repos/{encoded_name}/contents"
    )

    if not isinstance(root_contents, list):
        root_contents = []

    entries = [
        {
            "name": item["name"],
            "type": item["type"]
        }
        for item in root_contents
    ]

    return {
        "full_name": repo["full_name"],
        "description": repo.get("description"),
        "primary_language": repo.get("language"),
        "topics": repo.get("topics", []),
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "default_branch": repo.get("default_branch"),
        "homepage": repo.get("homepage"),
        "repository_url": repo["html_url"],
        "root_entries": entries,
        "note": (
            "root_entries 只包含仓库根目录。"
            "未读取子目录代码、依赖文件或完整源码，"
            "不能据此断言完整技术栈或内部实现。"
        )
    }
