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


    print("任务类型:", result["task_type"])
    print("置信度:", result["confidence"])
    print("分析:", result["analysis_result"])
    print("回答:", result["final_answer"])
    
if __name__ == "__main__":
    main()