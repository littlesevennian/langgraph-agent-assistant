
import uuid

from langchain_core.messages import HumanMessage
from src.agent.graph import graph


def main():
    thread_id = str(uuid.uuid4())

    print("=" * 50)
    print("Agent Assistant CLI Demo")
    print("输入 exit 退出，输入 new 开始新会话")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nUser: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Agent: 再见！")
                break

            if user_input.lower() == "new":
                thread_id = str(uuid.uuid4())
                print("Agent: 已开始新会话。")
                continue

            state = {
                "user_input": user_input,
                "messages": [
                    HumanMessage(content=user_input)
                ]
            }

            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }

            result = graph.invoke(
                state,
                config=config
            )

            answer = result.get("final_answer")

            if not answer:
                answer = "本次执行未生成最终回答，请检查节点输出。"

            print(f"\nAgent: {answer}")

        except (KeyboardInterrupt, EOFError):
            print("\nAgent: 已退出。")
            break

        except Exception as e:
            print(f"\nAgent运行失败: {e}")


if __name__ == "__main__":
    main()