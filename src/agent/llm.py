import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .tools.github_tool import github_analyzer


load_dotenv()


api_key = os.getenv("API_KEY")

base_url = os.getenv("BASE_URL")

model_name = os.getenv(
    "MODEL_NAME",
    "qwen3.7-max"
)

temperature = float(
    os.getenv(
        "TEMPERATURE",
        "0.7"
    )
)


llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.7
)


llm_with_tools = llm.bind_tools(
    [
        github_analyzer
    ]
)