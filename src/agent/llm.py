import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .tools.github_tool import github_analyzer


load_dotenv()


api_key = os.getenv( "LLM_API_KEY")

base_url = os.getenv("LLM_BASE_URL")

model_name = os.getenv(
    "LLM_MODEL",
    "qwen3.7-max"
)

temperature = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.7"
    )
)

if not api_key:
    raise ValueError(
        "LLM_API_KEY is missing"
    )


if not base_url:
    raise ValueError(
        "LLM_BASE_URL is missing"
    )


llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=temperature
)


llm_with_tools = llm.bind_tools(
    [
        github_analyzer
    ]
)
