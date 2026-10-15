import unittest
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.agent.memory import profile_store


class CheckpointMemoryTests(unittest.TestCase):


    def test_01_profile_shared_between_threads(self):
        """
        同一个用户不同thread可以读取长期Memory
        """

        with tempfile.TemporaryDirectory() as tmp:

            old_path = profile_store.PROFILE_PATH

            profile_store.PROFILE_PATH = (
                str(Path(tmp) / "profile.json")
            )

            try:

                profile_store.save_profile(
                    {
                        "career":
                        "嵌入式工程师"
                    }
                )


                profile = (
                    profile_store.load_profile()
                )


                self.assertEqual(
                    profile["career"],
                    "嵌入式工程师"
                )


            finally:

                profile_store.PROFILE_PATH = old_path



    def test_02_thread_state_should_not_equal_memory(self):

        """
        验证：
        Thread State 和 Memory不是同一个东西
        """

        thread_state = {
            "thread_id":"thread_001",
            "messages":[
                HumanMessage(
                    content="你好"
                )
            ]
        }


        profile = {
            "career":
            "嵌入式工程师"
        }


        self.assertNotEqual(
            thread_state.get("thread_id"),
            profile.get("career")
        )



    def test_03_profile_survive_new_thread(self):

        """
        新thread仍然可以读取旧Memory
        """

        with tempfile.TemporaryDirectory() as tmp:

            old_path = profile_store.PROFILE_PATH

            profile_store.PROFILE_PATH = (
                str(Path(tmp) / "profile.json")
            )


            try:

                # thread_001产生记忆

                profile_store.save_profile(
                    {
                        "skills":[
                            "STM32"
                        ]
                    }
                )


                # thread_002读取

                profile = (
                    profile_store.load_profile()
                )


                self.assertIn(
                    "STM32",
                    profile["skills"]
                )


            finally:

                profile_store.PROFILE_PATH = old_path



if __name__ == "__main__":
    unittest.main()