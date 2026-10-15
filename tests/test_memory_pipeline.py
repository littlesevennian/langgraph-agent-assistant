import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agent.nodes.memory_trigger import memory_trigger_node
from src.agent.nodes.profile_extractor import profile_extractor_node
from src.agent.memory import profile_store


class MemoryPipelineTests(unittest.TestCase):

    def test_01_memory_update_pipeline_success(self):
        """
        测试：
        用户信息 -> Memory Trigger -> Profile Extractor -> Storage
        """

        trigger_response = MagicMock()

        trigger_response.content = """
        {
            "need_memory_update": true,
            "memory_types": [
                "career",
                "skill"
            ],
            "confidence": 0.95,
            "reason": "用户提供职业和技能信息"
        }
        """


        extractor_response = MagicMock()

        extractor_response.content = """
        {
            "current_status": {
                "career": "嵌入式工程师"
            },
            "skills": [
                "STM32",
                "FreeRTOS"
            ]
        }
        """


        state = {
            "user_input":
                "我是嵌入式工程师，会STM32和FreeRTOS"
        }


        with tempfile.TemporaryDirectory() as tmp:

            old_path = profile_store.PROFILE_PATH

            profile_store.PROFILE_PATH = (
                str(Path(tmp) / "profile.json")
            )

            try:

                with patch(
                    "src.agent.nodes.memory_trigger.invoke_llm",
                    return_value=trigger_response
                ):

                    trigger_result = memory_trigger_node(
                        state
                    )


                self.assertTrue(
                    trigger_result["memory_decision"]
                    ["need_memory_update"]
                )


                with patch(
                    "src.agent.nodes.profile_extractor.invoke_llm",
                    return_value=extractor_response
                ):

                    profile_extractor_node(
                        state
                    )


                profile = profile_store.load_profile()


                self.assertEqual(
                    profile["current_status"]["career"],
                    "嵌入式工程师"
                )


                self.assertIn(
                    "STM32",
                    profile["skills"]
                )


            finally:

                profile_store.PROFILE_PATH = old_path



    def test_02_no_memory_update_skip_profile(self):
        """
        测试：
        普通知识问题不应该写入Memory
        """

        trigger_response = MagicMock()

        trigger_response.content = """
        {
            "need_memory_update": false,
            "memory_types": [
                "none"
            ],
            "confidence": 0.95,
            "reason": "普通学习问题"
        }
        """


        state = {
            "user_input":
                "什么是LangGraph"
        }


        with patch(
            "src.agent.nodes.memory_trigger.invoke_llm",
            return_value=trigger_response
        ):

            result = memory_trigger_node(
                state
            )


        self.assertFalse(
            result["memory_decision"]
            ["need_memory_update"]
        )



    def test_03_profile_save_failure_not_crash(self):

        """
        测试：
        Memory保存失败边界
        """

        extractor_response = MagicMock()

        extractor_response.content = """
        {
            "skills":[
                "STM32"
            ]
        }
        """


        state = {
            "user_input":
                "我会STM32"
        }


        with patch(
            "src.agent.nodes.profile_extractor.invoke_llm",
            return_value=extractor_response
        ):

            with patch(
                "src.agent.nodes.profile_extractor.save_profile",
                side_effect=Exception(
                    "disk error"
                )
            ):

                result = profile_extractor_node(
                    state
                )


        self.assertEqual(
            result,
            {}
        )



if __name__ == "__main__":
    unittest.main()