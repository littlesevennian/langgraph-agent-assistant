from src.agent.graph import graph


def main():
    user_input = input("请输入你的问题或目标：")

    initial_state = {
        "messages": [],
        "user_input": user_input,
        "task_type": "",
        "analysis_result": "",
        "final_answer": ""
    }

    result = graph.invoke(initial_state)

    print("\n===== 任务类型 =====")
    print(result["task_type"])

    print("\n===== 分析结果 =====")
    print(result["analysis_result"])

    print("\n===== 最终回答 =====")
    print(result["final_answer"])

    for msg in result["messages"]:
        print(msg.content)


if __name__ == "__main__":
    main()