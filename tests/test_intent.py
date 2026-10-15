"""Deterministic unit tests for src.agent.nodes.intent (no real LLM calls)."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.agent.nodes.intent import (
    IntentResult,
    classify_intent,
    intent_classifier_node,
)
from src.agent.utils.errors import AgentError, ErrorType


class IntentClassifierTests(unittest.TestCase):
    def setUp(self):
        self.user_input = "什么是异步IO？"
        self.messages = [HumanMessage(content=self.user_input)]

    @staticmethod
    def llm_response(payload):
        return SimpleNamespace(content=payload)

    def test_01_valid_json_returns_intent_result(self):
        payload = json.dumps({
            "task_type": "learning", "confidence": 0.91,
            "reason": "通用技术概念解释",
        }, ensure_ascii=False)
        with patch("src.agent.nodes.intent.invoke_llm", return_value=self.llm_response(payload)) as mocked:
            result = classify_intent(self.user_input, self.messages)
        self.assertIsInstance(result, IntentResult)
        self.assertEqual(result.task_type, "learning")
        self.assertEqual(result.confidence, 0.91)
        self.assertEqual(result.reason, "通用技术概念解释")
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.kwargs["caller"], "intent_classifier")
        self.assertIn(self.user_input, mocked.call_args.kwargs["input_data"])

    def test_02_invalid_json_raises_parse_error(self):
        with patch("src.agent.nodes.intent.invoke_llm", return_value=self.llm_response("task_type: learning")):
            with self.assertRaises(AgentError) as caught:
                classify_intent(self.user_input, self.messages)
        self.assertEqual(caught.exception.error_type, ErrorType.PARSE_ERROR)

    def test_03_invalid_task_type_raises_validation_error(self):
        payload = json.dumps({
            "task_type": "other", "confidence": 0.8,
            "reason": "非法枚举值",
        }, ensure_ascii=False)
        with patch("src.agent.nodes.intent.invoke_llm", return_value=self.llm_response(payload)):
            with self.assertRaises(AgentError) as caught:
                classify_intent(self.user_input, self.messages)
        self.assertEqual(caught.exception.error_type, ErrorType.VALIDATION_ERROR)

    def test_04_llm_agent_error_is_propagated(self):
        injected = AgentError(ErrorType.LLM_ERROR, "Injected LLM failure")
        with patch("src.agent.nodes.intent.invoke_llm", side_effect=injected):
            with self.assertRaises(AgentError) as caught:
                classify_intent(self.user_input, self.messages)
        self.assertIs(caught.exception, injected)

    def test_05_node_degrades_to_fallback_on_agent_error(self):
        injected = AgentError(ErrorType.PARSE_ERROR, "Injected parse failure")
        state = {"user_input": self.user_input, "messages": self.messages}
        with patch("src.agent.nodes.intent.classify_intent", side_effect=injected) as mocked:
            result = intent_classifier_node(state)
        self.assertEqual(result["task_type"], "fallback")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("fallback", result["analysis"])
        mocked.assert_called_once_with(user_input=self.user_input, messages=self.messages)

    def test_06_node_returns_successful_classification(self):
        expected = IntentResult(task_type="project", confidence=0.93, reason="具体项目架构分析")
        state = {"user_input": self.user_input, "messages": self.messages}
        with patch("src.agent.nodes.intent.classify_intent", return_value=expected) as mocked:
            result = intent_classifier_node(state)
        self.assertEqual(result, {
            "task_type": "project", "confidence": 0.93,
            "analysis": "具体项目架构分析",
        })
        mocked.assert_called_once_with(user_input=self.user_input, messages=self.messages)


if __name__ == "__main__":
    unittest.main()
