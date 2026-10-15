"""Deterministic routing integration tests using production routing functions.

Run: python -m unittest tests.test_graph_routing -v
No real LLM, GitHub, or profile storage calls are made.
"""
import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import State
from src.agent.graph import route_memory, route_task


class GraphRoutingTests(unittest.TestCase):
    def setUp(self):
        self.visited = []
        self.task_type = "resume"
        self.confidence = 0.95
        self.need_memory_update = False
        self.fail_classification = False

        def memory_trigger(state):
            self.visited.append("memory_trigger")
            return {"memory_decision": {"need_memory_update": self.need_memory_update}}

        def extractor(state):
            self.visited.append("profile_extractor")
            return {"user_profile": {"test_only": True}}

        def classifier(state):
            self.visited.append("intent_classifier")
            if self.fail_classification:
                # This is the node's documented degradation result, not a real LLM call.
                return {"task_type": "fallback", "confidence": 0.0, "analysis": "degraded"}
            return {"task_type": self.task_type, "confidence": self.confidence, "analysis": "mock"}

        def endpoint(name):
            def run(state):
                self.visited.append(name)
                return {"final_answer": name}
            return run

        # Rebuild the production topology with harmless stubs. The routing
        # functions themselves are imported from the real graph module.
        wf = StateGraph(State)
        wf.add_node("memory_trigger", memory_trigger)
        wf.add_node("profile_extractor", extractor)
        wf.add_node("intent_classifier", classifier)

        destinations = [
            "resume_response",
            "project_agent",
            "learning_response",
            "memory_query",
            "fallback_response",
        ]

        # 1. 注册五个业务节点
        for name in destinations:
            wf.add_node(name, endpoint(name))

        # 2. 设置图的入口
        wf.add_edge(START, "memory_trigger")

        # 3. 注册记忆判断条件边
        wf.add_conditional_edges(
            "memory_trigger",
            route_memory,
            {
                "profile_extractor": "profile_extractor",
                "intent_classifier": "intent_classifier",
            },
        )

        # 4. 记忆提取完成后进入意图分类
        wf.add_edge(
            "profile_extractor",
            "intent_classifier",
        )

        # 5. 注册意图分类条件边
        wf.add_conditional_edges(
            "intent_classifier",
            route_task,
            {name: name for name in destinations},
        )

        # 6. 五个业务节点执行完毕后结束
        for name in destinations:
            wf.add_edge(name, END)

        # 7. 编译测试图
        self.graph = wf.compile(checkpointer=MemorySaver())

    def invoke(self):
        return self.graph.invoke(
            {"user_input": "unit test", "messages": [HumanMessage(content="unit test")]},
            config={"configurable": {"thread_id": str(uuid4())}},
        )

    def test_01_five_intents_route_to_correct_nodes(self):
        cases = {
            "resume": "resume_response", "project": "project_agent",
            "learning": "learning_response", "memory_query": "memory_query",
            "fallback": "fallback_response",
        }
        for task_type, target in cases.items():
            with self.subTest(task_type=task_type):
                self.visited.clear()
                self.task_type = task_type
                state = self.invoke()
                self.assertEqual(state["final_answer"], target)
                self.assertEqual(self.visited, ["memory_trigger", "intent_classifier", target])

    def test_02_classifier_degradation_routes_to_fallback(self):
        self.fail_classification = True
        state = self.invoke()
        self.assertEqual(state["final_answer"], "fallback_response")
        self.assertEqual(state["confidence"], 0.0)

    def test_03_low_confidence_routes_to_fallback(self):
        self.task_type = "project"
        self.confidence = 0.79
        state = self.invoke()
        self.assertEqual(state["final_answer"], "fallback_response")

    def test_04_memory_update_visits_extractor_before_classifier(self):
        self.need_memory_update = True
        state = self.invoke()
        self.assertEqual(state["final_answer"], "resume_response")
        self.assertEqual(self.visited, ["memory_trigger", "profile_extractor", "intent_classifier", "resume_response"])
        self.assertEqual(state["user_profile"], {"test_only": True})

    def test_05_unknown_intent_routes_to_fallback(self):
        self.task_type = "unknown"
        state = self.invoke()
        self.assertEqual(state["final_answer"], "fallback_response")


if __name__ == "__main__":
    unittest.main()
