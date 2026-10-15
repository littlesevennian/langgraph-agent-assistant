
import unittest
from unittest.mock import patch

from src.agent.tools.github_tool import github_analyzer


class GitHubToolTests(unittest.TestCase):

    @patch("src.agent.tools.github_tool.github_get")
    def test_01_successful_repository_query(self, mock_get):
        """
        测试1：
        模拟GitHub API正常响应。
        验证工具能正确解析仓库信息及根目录。
        """

        mock_get.side_effect = [
            {
                "full_name": "react/react",
                "description": "React library",
                "language": "JavaScript",
                "topics": ["react", "javascript"],
                "stargazers_count": 1000,
                "forks_count": 200,
                "default_branch": "main",
                "homepage": "https://react.dev",
                "html_url": "https://github.com/react/react",
            },
            [
                {
                    "name": "packages",
                    "type": "dir"
                },
                {
                    "name": "README.md",
                    "type": "file"
                }
            ]
        ]

        result = github_analyzer.invoke(
            {"repo_name": "facebook/react"}
        )

        self.assertEqual(
            result["full_name"],
            "react/react"
        )

        self.assertEqual(
            result["primary_language"],
            "JavaScript"
        )

        self.assertEqual(
            result["stars"],
            1000
        )

        self.assertEqual(
            result["forks"],
            200
        )

        self.assertEqual(
            result["root_entries"],
            [
                {"name": "packages", "type": "dir"},
                {"name": "README.md", "type": "file"}
            ]
        )

        self.assertEqual(
            mock_get.call_count,
            2
        )

        mock_get.assert_any_call(
            "/repos/facebook/react"
        )

        mock_get.assert_any_call(
            "/repos/facebook/react/contents"
        )

    @patch("src.agent.tools.github_tool.github_get")
    def test_02_api_failure(self, mock_get):
        """
        测试2：
        模拟GitHub API返回404。
        验证工具不会伪造仓库数据。
        """

        mock_get.side_effect = RuntimeError(
            "GitHub API HTTP 404"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "GitHub API HTTP 404"
        ):
            github_analyzer.invoke(
                {"repo_name": "invalid/repository"}
            )

        mock_get.assert_called_once_with(
            "/repos/invalid/repository"
        )


if __name__ == "__main__":
    unittest.main()
