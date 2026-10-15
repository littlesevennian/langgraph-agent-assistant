import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agent.nodes.profile_extractor import merge_profile
from src.agent.memory import profile_store
from src.agent.nodes.memory_trigger import memory_trigger_node
from src.agent.utils.errors import AgentError, ErrorType


class MemoryTests(unittest.TestCase):

    # =========================
    # merge_profile tests
    # =========================

    def test_01_merge_list_without_duplicate(self):
        old = {
            "skills": [
                "STM32"
            ]
        }

        new = {
            "skills": [
                "STM32",
                "FreeRTOS"
            ]
        }

        result = merge_profile(old, new)

        self.assertEqual(
            result["skills"],
            [
                "STM32",
                "FreeRTOS"
            ]
        )


    def test_02_merge_new_field(self):
        old = {
            "skills": []
        }

        new = {
            "projects": [
                "IoT Gateway"
            ]
        }

        result = merge_profile(old, new)

        self.assertEqual(
            result["projects"],
            [
                "IoT Gateway"
            ]
        )


    def test_03_none_should_not_override_old_value(self):
        old = {
            "current_status": {
                "career": "嵌入式工程师"
            }
        }

        new = {
            "current_status": {
                "career": None
            }
        }

        result = merge_profile(old, new)

        self.assertEqual(
            result["current_status"]["career"],
            "嵌入式工程师"
        )


    # =========================
    # profile_store tests
    # =========================

    def test_04_load_non_exist_profile(self):

        with tempfile.TemporaryDirectory() as tmp:

            old_path = profile_store.PROFILE_PATH

            profile_store.PROFILE_PATH = (
                str(Path(tmp) / "profile.json")
            )

            try:
                result = profile_store.load_profile()

                self.assertEqual(
                    result,
                    {}
                )

            finally:
                profile_store.PROFILE_PATH = old_path


    def test_05_save_and_load_profile(self):

        with tempfile.TemporaryDirectory() as tmp:

            old_path = profile_store.PROFILE_PATH

            profile_store.PROFILE_PATH = (
                str(Path(tmp) / "profile.json")
            )

            try:

                data = {
                    "name": "张三"
                }

                profile_store.save_profile(data)

                result = profile_store.load_profile()

                self.assertEqual(
                    result,
                    data
                )

            finally:
                profile_store.PROFILE_PATH = old_path


    # =========================
    # memory trigger tests
    # =========================

    def test_06_memory_trigger_json_error_degrade(self):

        fake_response = MagicMock()
        fake_response.content = "invalid json"


        state = {
            "user_input": "我是嵌入式工程师",
        }


        with patch(
            "src.agent.nodes.memory_trigger.invoke_llm",
            return_value=fake_response
        ):

            result = memory_trigger_node(state)


        self.assertFalse(
            result["memory_decision"]["need_memory_update"]
        )

        self.assertEqual(
            result["memory_decision"]["memory_types"],
            ["none"]
        )


if __name__ == "__main__":
    unittest.main()