import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agent.nodes.profile_extractor import profile_extractor_node
from src.agent.memory import profile_store


class ProfileExtractorTests(unittest.TestCase):

    def test_01_valid_profile_update_saved(self):
        """
        正常Profile提取
        """

        fake_response = MagicMock()

        fake_response.content = """
        {
            "basic_info": {
                "name": "张三"
            },
            "skills": [
                "STM32",
                "FreeRTOS"
            ]
        }
        """

        state = {
            "user_input": "我是嵌入式工程师，会STM32和FreeRTOS"
        }


        with tempfile.TemporaryDirectory() as tmp:

            old_path = profile_store.PROFILE_PATH

            profile_store.PROFILE_PATH = (
                str(Path(tmp) / "profile.json")
            )

            try:

                with patch(
                    "src.agent.nodes.profile_extractor.invoke_llm",
                    return_value=fake_response
                ):

                    result = profile_extractor_node(state)


                profile = profile_store.load_profile()


                self.assertEqual(
                    profile["basic_info"]["name"],
                    "张三"
                )

                self.assertIn(
                    "STM32",
                    profile["skills"]
                )


            finally:
                profile_store.PROFILE_PATH = old_path



    def test_02_invalid_profile_schema_rejected(self):

        """
        Schema校验失败
        """

        fake_response = MagicMock()

        fake_response.content = """
        {
            "skills": "STM32"
        }
        """


        state = {
            "user_input": "测试错误profile"
        }


        with patch(
            "src.agent.nodes.profile_extractor.invoke_llm",
            return_value=fake_response
        ):

            result = profile_extractor_node(state)


        self.assertEqual(
            result,
            {}
        )



    def test_03_duplicate_skill_should_not_duplicate(self):

        """
        重复技能不会无限增加
        """

        old_profile = {
            "skills":[
                "STM32"
            ]
        }


        fake_response = MagicMock()

        fake_response.content = """
        {
            "skills":[
                "STM32"
            ]
        }
        """


        state = {
            "user_input":"重复技能测试"
        }


        with tempfile.TemporaryDirectory() as tmp:

            old_path = profile_store.PROFILE_PATH

            profile_store.PROFILE_PATH = (
                str(Path(tmp) / "profile.json")
            )

            try:

                profile_store.save_profile(old_profile)


                with patch(
                    "src.agent.nodes.profile_extractor.invoke_llm",
                    return_value=fake_response
                ):

                    profile_extractor_node(state)


                profile = profile_store.load_profile()


                self.assertEqual(
                    profile["skills"].count("STM32"),
                    1
                )


            finally:
                profile_store.PROFILE_PATH = old_path



    def test_04_llm_invalid_json_should_not_crash(self):

        """
        LLM返回非法JSON
        """

        fake_response = MagicMock()

        fake_response.content = """
        hello world
        """


        state = {
            "user_input":"测试非法JSON"
        }


        with patch(
            "src.agent.nodes.profile_extractor.invoke_llm",
            return_value=fake_response
        ):

            result = profile_extractor_node(state)

        self.assertEqual(
            result,
            {}
        )


if __name__ == "__main__":
    unittest.main()