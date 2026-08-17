from src.agent.graph import graph

from langchain_core.messages import HumanMessage


def main():
    config = {
            "configurable": {
                "thread_id": "user_001"
            }
        }

     
    while True:

        user_input = input(
            "请输入:"
        )

        if user_input == "exit":
            break


        initial_state = {
            "user_input": user_input,
            "messages": [
                HumanMessage(content=user_input)
            ],
            "task_type":"",
            "confidence":0,
            "analysis_result":"",
            "final_answer":""
        }

        
        result = graph.invoke(
            initial_state,
            config=config
        )
       
        print("任务类型:", result["task_type"])
        print("置信度:", result["confidence"])
        print("分析:", result["analysis"])
        print("回答:", result["final_answer"])
        print(result["messages"])
   
    
if __name__ == "__main__":
    main()