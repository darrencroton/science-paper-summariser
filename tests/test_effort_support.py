import unittest
from unittest.mock import patch

from providers.cli import ClaudeCLI, CodexCLI, CopilotCLI, GeminiCLI, OpenCodeCLI
from providers import create_provider
from summarise import build_provider_config, parse_cli_args, validate_startup_selection


class ParseCliArgsTests(unittest.TestCase):
    def test_defaults_without_args(self):
        self.assertEqual(parse_cli_args([]), ("cli", "claude", None, None))

    def test_allows_effort_with_default_mode_and_provider(self):
        self.assertEqual(parse_cli_args(["--effort", "high"]), ("cli", "claude", None, "high"))

    def test_allows_effort_before_mode_and_provider(self):
        self.assertEqual(
            parse_cli_args(["--effort", "medium", "cli", "claude"]),
            ("cli", "claude", None, "medium"),
        )

    def test_allows_effort_with_model_override(self):
        self.assertEqual(
            parse_cli_args(["cli", "codex", "gpt-5.4", "--effort", "low"]),
            ("cli", "codex", "gpt-5.4", "low"),
        )

    def test_rejects_invalid_effort(self):
        with self.assertRaisesRegex(ValueError, "Invalid effort"):
            parse_cli_args(["cli", "claude", "--effort", "xhigh"])

    def test_rejects_effort_in_api_mode(self):
        with self.assertRaisesRegex(ValueError, "only supported in cli mode"):
            parse_cli_args(["api", "openai", "gpt-5.2", "--effort", "high"])

    def test_build_provider_config_includes_effort_when_set(self):
        self.assertEqual(
            build_provider_config(model_override="gpt-5.4", effort="medium"),
            {"model": "gpt-5.4", "effort": "medium"},
        )

    @patch("summarise.create_provider")
    def test_validate_startup_selection_passes_effort_to_provider_creation(self, mock_create_provider):
        mock_provider = object()
        mock_create_provider.return_value = mock_provider

        result = validate_startup_selection(["cli", "codex", "gpt-5.4", "--effort", "high"])

        mock_create_provider.assert_called_once_with(
            "cli",
            "codex",
            config={"model": "gpt-5.4", "effort": "high"},
        )
        self.assertEqual(result, ("cli", "codex", "gpt-5.4", "high", mock_provider))


class CliProviderEffortTests(unittest.TestCase):
    @patch("providers.cli.shutil.which", return_value="/usr/bin/claude")
    def test_claude_cli_builds_command_with_effort(self, _mock_which):
        provider = ClaudeCLI({"model": "claude-sonnet-4-6", "effort": "high"})

        self.assertEqual(
            provider._build_command("prompt"),
            [
                "claude",
                "--output-format",
                "text",
                "--model",
                "claude-sonnet-4-6",
                "--effort",
                "high",
                "-p",
                "prompt",
            ],
        )

    @patch("providers.cli.shutil.which", return_value="/usr/bin/codex")
    def test_codex_cli_builds_command_with_effort(self, _mock_which):
        provider = CodexCLI({"model": "gpt-5.4", "effort": "medium"})

        self.assertEqual(
            provider._build_command("prompt"),
            [
                "codex",
                "exec",
                "-c",
                'model="gpt-5.4"',
                "-c",
                'model_reasoning_effort="medium"',
                "-",
            ],
        )

    @patch("providers.cli.shutil.which", return_value="/usr/bin/copilot")
    def test_copilot_cli_builds_command_with_effort(self, _mock_which):
        provider = CopilotCLI({"model": "gpt-5.2", "effort": "low"})

        self.assertEqual(
            provider._build_command("prompt"),
            [
                "copilot",
                "--allow-all-tools",
                "--output-format",
                "text",
                "--silent",
                "--model",
                "gpt-5.2",
                "--effort",
                "low",
                "-p",
                "prompt",
            ],
        )

    @patch("providers.cli.shutil.which", return_value="/usr/bin/gemini")
    def test_gemini_cli_warns_and_ignores_effort(self, _mock_which):
        with self.assertLogs("providers.cli", level="WARNING") as captured_logs:
            provider = GeminiCLI({"effort": "high"})

        self.assertEqual(provider._build_command("prompt"), ["gemini", "-o", "text", "-p", "prompt"])
        self.assertEqual(
            captured_logs.output,
            ["WARNING:providers.cli:[WARNING] Gemini CLI ignores --effort; using Gemini defaults."],
        )

    @patch("providers.shutil.which", return_value="/usr/bin/claude")
    @patch("providers.cli.shutil.which", return_value="/usr/bin/claude")
    def test_create_provider_preserves_cli_effort_in_config(self, _mock_cli_which, _mock_provider_which):
        provider = create_provider("cli", "claude", config={"effort": "high"})

        self.assertEqual(provider.config["effort"], "high")
        self.assertEqual(provider.effort, "high")


class OpenCodeCLITests(unittest.TestCase):
    @patch("providers.cli.shutil.which", return_value="/usr/bin/opencode")
    def test_opencode_builds_command_without_model_or_effort(self, _mock_which):
        provider = OpenCodeCLI({})

        self.assertEqual(
            provider._build_command("prompt"),
            ["opencode", "run", "--format", "json", "prompt"],
        )

    @patch("providers.cli.shutil.which", return_value="/usr/bin/opencode")
    def test_opencode_builds_command_with_model(self, _mock_which):
        provider = OpenCodeCLI({"model": "ollama/llama3.2"})

        self.assertEqual(
            provider._build_command("prompt"),
            ["opencode", "run", "--format", "json", "--model", "ollama/llama3.2", "prompt"],
        )

    @patch("providers.cli.shutil.which", return_value="/usr/bin/opencode")
    def test_opencode_builds_command_with_effort(self, _mock_which):
        provider = OpenCodeCLI({"effort": "high"})

        self.assertEqual(
            provider._build_command("prompt"),
            ["opencode", "run", "--format", "json", "--variant", "high", "prompt"],
        )

    def test_opencode_extracts_text_from_json_events(self):
        raw = "\n".join(
            [
                '{"type":"step_start","part":{"type":"step-start"}}',
                '{"type":"text","part":{"type":"text","text":"# Title\\n"}}',
                '{"type":"text","part":{"type":"text","text":"Body"}}',
                '{"type":"step_finish","part":{"type":"step-finish"}}',
            ]
        )

        self.assertEqual(OpenCodeCLI._extract_text_from_json_events(raw), "# Title\nBody")

    def test_opencode_rejects_malformed_json_events(self):
        with self.assertRaisesRegex(ValueError, "malformed JSON event"):
            OpenCodeCLI._extract_text_from_json_events('{"type":"text"')

    def test_opencode_rejects_json_without_text_events(self):
        with self.assertRaisesRegex(ValueError, "contained no text events"):
            OpenCodeCLI._extract_text_from_json_events(
                '{"type":"step_finish","part":{"type":"step-finish"}}'
            )

    @patch("providers.shutil.which", return_value="/usr/bin/opencode")
    @patch("providers.cli.shutil.which", return_value="/usr/bin/opencode")
    def test_create_provider_registers_opencode(self, _mock_cli_which, _mock_provider_which):
        provider = create_provider("cli", "opencode")

        self.assertEqual(provider.provider_name, "opencode")
        self.assertEqual(provider.mode, "cli")


if __name__ == "__main__":
    unittest.main()
