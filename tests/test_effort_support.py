import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from providers.cli import ClaudeCLI, CodexCLI, CopilotCLI, GeminiCLI, OpenCodeCLI
from providers import create_provider
from summarise import (
    _call_llm_with_retry,
    build_provider_config,
    extract_arxiv_categories_from_api_xml,
    extract_arxiv_categories_from_html,
    filter_keywords_for_categories,
    build_fallback_tags,
    generate_tags,
    fit_prompt_to_provider_budget,
    insert_section,
    normalise_extracted_text,
    normalise_tags_section,
    parse_cli_args,
    process_file,
    SourceMetadata,
    GLOSSARY_MAX_TERMS,
    validate_glossary_section,
    validate_startup_selection,
    validate_tags_section,
)


class DummyBudgetedProvider:
    mode = "api"
    provider_name = "any-provider"

    def __init__(self, max_prompt_chars=None):
        self.model = None
        self.config = {}
        if max_prompt_chars is not None:
            self.config["max_prompt_chars"] = max_prompt_chars

    def supports_direct_pdf(self):
        return False


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
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        result = validate_startup_selection(["cli", "codex", "gpt-5.4", "--effort", "high"])

        mock_create_provider.assert_called_once_with(
            "cli",
            "codex",
            config={"model": "gpt-5.4", "effort": "high"},
        )
        self.assertEqual(result, ("cli", "codex", "gpt-5.4", "high", mock_provider))


class PromptHardeningTests(unittest.TestCase):
    def test_normalise_extracted_text_removes_pathological_table_rows(self):
        wide_table_row = "|" + (" noisy cell |" * 120)
        text = "\n".join(
            [
                "The paper reports a compact quiescent galaxy.",
                "| Filter | Value |",
                "| F150W | 1.23 |",
                wide_table_row,
                "The discussion remains available for exact quotes.",
            ]
        )

        cleaned = normalise_extracted_text(text)

        self.assertIn("| Filter | Value |", cleaned)
        self.assertIn("| F150W | 1.23 |", cleaned)
        self.assertNotIn(wide_table_row, cleaned)
        self.assertIn("The discussion remains available", cleaned)

    def test_normalise_extracted_text_removes_long_non_table_noise_lines(self):
        noisy_line = "<br>".join(["1.23+0.04"] * 140)
        text = "\n".join(
            [
                "The abstract remains intact.",
                noisy_line,
                "The conclusion remains intact.",
            ]
        )

        cleaned = normalise_extracted_text(text)

        self.assertIn("The abstract remains intact.", cleaned)
        self.assertNotIn(noisy_line, cleaned)
        self.assertIn("The conclusion remains intact.", cleaned)

    def test_fit_prompt_to_provider_budget_drops_references_before_appendix(self):
        provider = DummyBudgetedProvider(max_prompt_chars=1200)
        system_prompt = "system"
        template = "template"
        paper_text = "\n".join(
            [
                "# Paper",
                "Main science result.",
                "#### REFERENCES",
                "Reference noise. " * 200,
                "# APPENDIX",
                "Appendix content that should not be reached.",
            ]
        )

        with self.assertLogs(level="WARNING"):
            reduced_text, user_prompt = fit_prompt_to_provider_budget(
                provider,
                system_prompt,
                paper_text,
                template,
            )

        self.assertIn("Main science result.", reduced_text)
        self.assertNotIn("Reference noise.", reduced_text)
        self.assertNotIn("Appendix content", reduced_text)
        self.assertIn("Main science result.", user_prompt)

    def test_fit_prompt_to_provider_budget_applies_to_any_provider(self):
        provider = DummyBudgetedProvider(max_prompt_chars=1200)
        system_prompt = "system"
        template = "template"
        paper_text = "\n".join(
            [
                "# Paper",
                "Main science result.",
                "#### REFERENCES",
                "Reference noise. " * 200,
            ]
        )

        with self.assertLogs(level="WARNING"):
            reduced_text, _user_prompt = fit_prompt_to_provider_budget(
                provider,
                system_prompt,
                paper_text,
                template,
            )

        self.assertNotIn("Reference noise.", reduced_text)

    def test_fit_prompt_to_provider_budget_drops_markdown_references_heading(self):
        provider = DummyBudgetedProvider(max_prompt_chars=1200)
        system_prompt = "system"
        template = "template"
        paper_text = "\n".join(
            [
                "# Paper",
                "Main science result.",
                "## References",
                "Reference noise. " * 200,
            ]
        )

        with self.assertLogs(level="WARNING"):
            reduced_text, user_prompt = fit_prompt_to_provider_budget(
                provider,
                system_prompt,
                paper_text,
                template,
            )

        self.assertIn("Main science result.", reduced_text)
        self.assertNotIn("Reference noise.", reduced_text)
        self.assertIn("Main science result.", user_prompt)

    def test_fit_prompt_to_provider_budget_drops_singular_reference_heading(self):
        provider = DummyBudgetedProvider(max_prompt_chars=1200)
        system_prompt = "system"
        template = "template"
        paper_text = "\n".join(
            [
                "# Paper",
                "Main science result.",
                "## Reference",
                "Reference noise. " * 200,
            ]
        )

        with self.assertLogs(level="WARNING"):
            reduced_text, _user_prompt = fit_prompt_to_provider_budget(
                provider,
                system_prompt,
                paper_text,
                template,
            )

        self.assertIn("Main science result.", reduced_text)
        self.assertNotIn("Reference noise.", reduced_text)

    def test_fit_prompt_to_provider_budget_drops_references_and_notes_heading(self):
        provider = DummyBudgetedProvider(max_prompt_chars=1200)
        system_prompt = "system"
        template = "template"
        paper_text = "\n".join(
            [
                "# Paper",
                "Main science result.",
                "### References and Notes",
                "Reference note noise. " * 200,
            ]
        )

        with self.assertLogs(level="WARNING"):
            reduced_text, _user_prompt = fit_prompt_to_provider_budget(
                provider,
                system_prompt,
                paper_text,
                template,
            )

        self.assertIn("Main science result.", reduced_text)
        self.assertNotIn("Reference note noise.", reduced_text)

    def test_fit_prompt_to_provider_budget_drops_numbered_appendix_heading(self):
        provider = DummyBudgetedProvider(max_prompt_chars=1200)
        system_prompt = "system"
        template = "template"
        paper_text = "\n".join(
            [
                "# Paper",
                "Main science result.",
                "# Appendix A: Supporting Material",
                "Appendix noise. " * 200,
            ]
        )

        with self.assertLogs(level="WARNING"):
            reduced_text, _user_prompt = fit_prompt_to_provider_budget(
                provider,
                system_prompt,
                paper_text,
                template,
            )

        self.assertIn("Main science result.", reduced_text)
        self.assertNotIn("Appendix noise.", reduced_text)

    def test_process_file_sends_reduced_prompt_to_provider(self):
        provider = DummyBudgetedProvider(max_prompt_chars=3000)
        paper_text = "\n".join(
            [
                "# Paper",
                "Main science result.",
                "## References",
                "Reference noise. " * 200,
            ]
        )

        with (
            patch("summarise.read_input_file", return_value=(paper_text, None)),
            patch("summarise._write_debug_prompt"),
            patch(
                "summarise._call_llm_with_retry",
                return_value="# Paper\n\nSummary",
            ) as mock_call,
            patch("summarise.strip_preamble", side_effect=lambda summary: summary),
            patch(
                "summarise.enforce_source_metadata",
                side_effect=lambda summary, source_metadata: summary,
            ),
            patch("summarise.validate_summary"),
            patch(
                "summarise.generate_glossary",
                return_value="## Glossary\n\n| Term | Definition |\n|---|---|\n| Term | Definition. |",
            ),
            patch("summarise.generate_tags", return_value="## Tags\n\n#JWST\n\n#Galaxies"),
            patch("summarise.extract_metadata", return_value=("Paper", ["Smith"], "2026")),
            patch("summarise.save_summary", return_value=Path("output/Paper.md")),
            patch("summarise.move_to_done", return_value=Path("processed/Paper.txt")),
            self.assertLogs(level="WARNING"),
        ):
            success, original_filename, error = process_file(
                Path("paper.txt"),
                keywords="keywords",
                template="template",
                provider=provider,
            )

        self.assertTrue(success)
        self.assertEqual(original_filename, "paper.txt")
        self.assertIsNone(error)
        sent_user_prompt = mock_call.call_args.args[4]
        self.assertIn("Main science result.", sent_user_prompt)
        self.assertNotIn("Reference noise.", sent_user_prompt)


class LocalModelOptimisationTests(unittest.TestCase):
    _KEYWORDS = "\n\n".join(
        [
            "GENERAL\n#General",
            "PHYSICAL DATA AND PROCESSES\n#BlackHolePhysics",
            "ASTRONOMICAL INSTRUMENTATION, METHODS AND TECHNIQUES\n#Telescopes",
            "ASTRONOMICAL DATABASES\n#Surveys",
            "GALAXIES\n#GalaxiesEvolution\n#GalaxiesHighRedshift",
            "COSMOLOGY\n#CosmologyObservations",
            "PLANETARY SYSTEMS\n#PlanetsAndSatellitesDetection",
        ]
    )

    def test_extract_arxiv_categories_from_html_reads_primary_and_all_categories(self):
        html = """
        <td class="tablecell subjects">
          <span class="primary-subject">Astrophysics of Galaxies (astro-ph.GA)</span>;
          Cosmology and Nongalactic Astrophysics (astro-ph.CO)
        </td>
        """

        primary, categories = extract_arxiv_categories_from_html(html)

        self.assertEqual(primary, "astro-ph.GA")
        self.assertEqual(categories, ("astro-ph.GA", "astro-ph.CO"))

    def test_extract_arxiv_categories_from_api_xml_reads_primary_and_all_categories(self):
        xml = """
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <arxiv:primary_category term="astro-ph.GA" />
            <category term="astro-ph.GA" />
            <category term="astro-ph.CO" />
          </entry>
        </feed>
        """

        primary, categories = extract_arxiv_categories_from_api_xml(xml)

        self.assertEqual(primary, "astro-ph.GA")
        self.assertEqual(categories, ("astro-ph.GA", "astro-ph.CO"))

    def test_filter_keywords_for_categories_keeps_relevant_sections(self):
        filtered = filter_keywords_for_categories(self._KEYWORDS, ("astro-ph.CO",))

        self.assertIn("COSMOLOGY", filtered)
        self.assertIn("#CosmologyObservations", filtered)
        self.assertIn("GALAXIES", filtered)
        self.assertIn("GENERAL", filtered)
        self.assertNotIn("PLANETARY SYSTEMS", filtered)

    def test_filter_keywords_for_unknown_category_falls_back_to_full_list(self):
        self.assertEqual(
            filter_keywords_for_categories(self._KEYWORDS, ("cond-mat.stat-mech",)),
            self._KEYWORDS,
        )

    def test_validate_glossary_section_accepts_markdown_table(self):
        validate_glossary_section(
            "## Glossary\n\n"
            "| Term | Definition |\n"
            "|---|---|\n"
            "| **Redshift** | Stretching of observed wavelength by cosmic expansion. |"
        )

    def test_validate_glossary_section_rejects_extra_content(self):
        with self.assertRaisesRegex(ValueError, "only a two-column table"):
            validate_glossary_section(
                "## Glossary\n\n"
                "| Term | Definition |\n"
                "|---|---|\n"
                "| **Redshift** | Stretching of observed wavelength by cosmic expansion. |\n\n"
                "Additional commentary."
            )

    def test_validate_glossary_section_rejects_extra_columns(self):
        with self.assertRaisesRegex(ValueError, "only a two-column table"):
            validate_glossary_section(
                "## Glossary\n\n"
                "| Term | Definition |\n"
                "|---|---|\n"
                "| **Redshift** | Meaning | Extra |"
            )

    def test_validate_glossary_section_rejects_too_many_terms(self):
        rows = "\n".join(
            f"| **Term {index}** | Definition. |"
            for index in range(GLOSSARY_MAX_TERMS + 1)
        )

        with self.assertRaisesRegex(ValueError, "no more than"):
            validate_glossary_section(
                "## Glossary\n\n"
                "| Term | Definition |\n"
                "|---|---|\n"
                f"{rows}"
            )

    def test_validate_tags_section_accepts_one_or_two_hashtag_lines(self):
        validate_tags_section(
            "## Tags\n\n#JWST #CEERS\n\n#GalaxiesHighRedshift #CosmologyObservations",
            self._KEYWORDS,
        )

        validate_tags_section("## Tags\n\n#JWST")

    def test_normalise_tags_section_accepts_labeled_comma_separated_lines(self):
        self.assertEqual(
            normalise_tags_section(
                "## Tags\n\n"
                "Proper nouns: #JWST, #CEERS\n\n"
                "- Science keywords: #GalaxiesHighRedshift, #CosmologyObservations",
                self._KEYWORDS,
            ),
            "## Tags\n\n#JWST #CEERS\n\n"
            "#GalaxiesHighRedshift #CosmologyObservations",
        )

    def test_normalise_tags_section_routes_keyword_tags_from_first_line_to_science_line(self):
        self.assertEqual(
            normalise_tags_section(
                "## Tags\n\n#Surveys #JWST\n\n#GalaxiesHighRedshift",
                self._KEYWORDS,
            ),
            "## Tags\n\n#JWST\n\n#Surveys #GalaxiesHighRedshift",
        )

    def test_normalise_tags_section_drops_unknown_science_line_tags(self):
        self.assertEqual(
            normalise_tags_section(
                "## Tags\n\n#JWST\n\n#MadeUpScienceTag #GalaxiesHighRedshift",
                self._KEYWORDS,
            ),
            "## Tags\n\n#JWST\n\n#GalaxiesHighRedshift",
        )

    def test_generate_tags_returns_normalised_section(self):
        class Provider:
            def get_preferred_max_tokens(self):
                return 100

            def process_document(self, **_kwargs):
                return (
                    "## Tags\n\n"
                    "Proper nouns: #JWST, #CEERS\n\n"
                    "Science keywords: #GalaxiesHighRedshift, #CosmologyObservations"
                )

        result = generate_tags(
            "# Summary\n\nJWST and CEERS measured high-redshift galaxies.",
            self._KEYWORDS,
            Provider(),
        )

        self.assertEqual(
            result,
            "## Tags\n\n#JWST #CEERS\n\n"
            "#GalaxiesHighRedshift #CosmologyObservations",
        )

    def test_normalise_tags_section_truncates_too_many_tags(self):
        self.assertEqual(
            normalise_tags_section(
                "## Tags\n\n#A #B #C #D #E #F\n\n#GalaxiesHighRedshift",
                self._KEYWORDS,
            ),
            "## Tags\n\n#A #B #C #D #E\n\n#GalaxiesHighRedshift",
        )

    def test_validate_tags_section_trusts_proper_noun_tags(self):
        validate_tags_section(
            "## Tags\n\n#JamesWebbSpaceTelescope\n\n#GalaxiesHighRedshift",
            self._KEYWORDS,
        )

    def test_validate_tags_section_accepts_compound_tag_parts_from_summary(self):
        validate_tags_section(
            "## Tags\n\n#BehrooziSMHM\n\n#GalaxiesHighRedshift",
            self._KEYWORDS,
        )

    def test_generate_tags_saves_best_effort_when_science_tag_is_not_in_keyword_list(self):
        class Provider:
            def get_preferred_max_tokens(self):
                return 100

            def process_document(self, **_kwargs):
                return "## Tags\n\n#JWST\n\n#MadeUpScienceTag #CosmologyObservations"

        result = generate_tags("# Summary\n\nJWST observes galaxies.", self._KEYWORDS, Provider())

        self.assertIn("#JWST", result)
        self.assertIn("#CosmologyObservations", result)
        self.assertNotIn("#MadeUpScienceTag", result)

    def test_generate_tags_keeps_proper_noun_tag_not_found_in_summary(self):
        class Provider:
            def get_preferred_max_tokens(self):
                return 100

            def process_document(self, **_kwargs):
                return "## Tags\n\n#RomanSpaceTelescope\n\n#CosmologyObservations"

        result = generate_tags(
            "# Summary\n\nThe James Webb Space Telescope observes galaxies.",
            self._KEYWORDS,
            Provider(),
        )

        self.assertIn("#RomanSpaceTelescope", result)

    def test_generate_tags_falls_back_when_response_has_no_parseable_tags(self):
        class Provider:
            def get_preferred_max_tokens(self):
                return 100

            def process_document(self, **_kwargs):
                return "## Tags\n\nNo useful tags."

        result = generate_tags(
            "# Summary\n\nCosmology observations of galaxies.",
            self._KEYWORDS,
            Provider(),
        )

        self.assertIn("#CosmologyObservations", result)

    def test_build_fallback_tags_derives_science_tags_from_summary(self):
        result = build_fallback_tags(
            "# Summary\n\nCosmology observations of galaxies.",
            self._KEYWORDS,
        )

        self.assertIn("#CosmologyObservations", result)

    def test_insert_section_places_generated_content_before_references(self):
        summary = "# Paper\n\n## Results\n\n- Result[^1]\n\n## References\n\n[^1]: \"quote\""
        result = insert_section(summary, "## Tags\n\n#JWST\n\n#Galaxies")

        self.assertLess(result.index("## Tags"), result.index("## References"))
        self.assertIn("## Results", result)

    @patch("summarise.interruptible_sleep", return_value=False)
    def test_call_llm_retries_failed_section_validation(self, _mock_sleep):
        call_count = 0

        class Provider:
            def get_preferred_max_tokens(self):
                return 100

            def process_document(self, **_kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "## Tags\n\nNo parseable tags"
                return "## Tags\n\n#JWST\n\n#CosmologyObservations"

        result = _call_llm_with_retry(
            Provider(),
            "",
            False,
            "system",
            "user",
            response_validator=validate_tags_section,
        )

        self.assertEqual(call_count, 2)
        self.assertIn("#CosmologyObservations", result)

    @patch("summarise.interruptible_sleep", return_value=False)
    def test_call_llm_raises_after_repeated_section_validation_failures(self, _mock_sleep):
        class Provider:
            def get_preferred_max_tokens(self):
                return 100

            def process_document(self, **_kwargs):
                return "## Glossary\n\nnot a table"

        with self.assertRaisesRegex(ValueError, "markdown table"):
            _call_llm_with_retry(
                Provider(),
                "",
                False,
                "system",
                "user",
                max_retries=2,
                response_validator=validate_glossary_section,
            )

    def test_process_file_generates_glossary_and_tags_after_main_summary(self):
        provider = DummyBudgetedProvider(max_prompt_chars=3000)
        paper_text = "# Paper\n\nMain science result."
        main_summary = (
            "# Paper\n\n"
            "Authors: Smith A.\n"
            "Published: January 2026 ([Link](https://arxiv.org/abs/2601.00001))\n\n"
            "## Results\n\n"
            "- Main science result[^1]\n\n"
            "## References\n\n"
            "[^1]: \"Main science result.\" (Abstract, p.1)\n"
        )

        captured = {}

        def fake_glossary(summary, _provider):
            captured["glossary_summary"] = summary
            return "## Glossary\n\n| Term | Definition |\n|---|---|\n| Result | A reported finding. |"

        def fake_tags(summary, keywords, _provider):
            captured["tags_summary"] = summary
            captured["tag_keywords"] = keywords
            return "## Tags\n\n#JWST\n\n#CosmologyObservations"

        with (
            patch("summarise.read_input_file", return_value=(paper_text, None)),
            patch("summarise._write_debug_prompt"),
            patch("summarise._call_llm_with_retry", return_value=main_summary) as mock_call,
            patch("summarise.strip_preamble", side_effect=lambda summary: summary),
            patch(
                "summarise.enforce_source_metadata",
                side_effect=lambda summary, source_metadata: summary,
            ),
            patch(
                "summarise.extract_source_metadata",
                return_value=SourceMetadata(
                    source_type="arxiv",
                    identifier="2601.00001",
                    canonical_url="https://arxiv.org/abs/2601.00001",
                    published_label="January 2026",
                    primary_category="astro-ph.CO",
                    categories=("astro-ph.CO",),
                ),
            ),
            patch("summarise.generate_glossary", side_effect=fake_glossary),
            patch("summarise.generate_tags", side_effect=fake_tags),
            patch("summarise.extract_metadata", return_value=("Paper", ["Smith"], "2026")),
            patch("summarise.save_summary", return_value=Path("output/Paper.md")) as mock_save,
            patch("summarise.move_to_done", return_value=Path("processed/Paper.txt")),
        ):
            success, _original_filename, error = process_file(
                Path("paper.txt"),
                keywords=self._KEYWORDS,
                template="## Results\n\n- Bullet with footnote\n\n## References\n\n[^1]: quote",
                provider=provider,
            )

        self.assertTrue(success)
        self.assertIsNone(error)
        sent_user_prompt = mock_call.call_args.args[4]
        self.assertNotIn("## Glossary", sent_user_prompt)
        self.assertNotIn("## Tags", sent_user_prompt)
        self.assertNotIn("#CosmologyObservations", sent_user_prompt)
        self.assertIn("Main science result", captured["glossary_summary"])
        self.assertNotIn("## Glossary", captured["tags_summary"])
        self.assertIn("#CosmologyObservations", captured["tag_keywords"])
        saved_summary = mock_save.call_args.args[0]
        self.assertLess(saved_summary.index("## Glossary"), saved_summary.index("## Tags"))
        self.assertLess(saved_summary.index("## Tags"), saved_summary.index("## References"))


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


class ReferencesEnforcementTests(unittest.TestCase):
    """_call_llm_with_retry retries when [^N] markers are present but ## References is absent."""

    # Minimal summary that contains an inline footnote marker but no References section.
    _BAD = "# Title\n\nAuthors: Smith A.\n\n[^1] key finding\n\n## Glossary\n| Term | Def |\n"
    # Same summary with the References section appended — should be accepted.
    _GOOD = _BAD + '\n## References\n\n[^1]: "exact quote" (Section 1, p.1)\n'

    class _StubProvider:
        def get_preferred_max_tokens(self):
            return 100

        def process_document(self, **_kwargs):
            raise NotImplementedError("override per test")

    @patch("summarise.interruptible_sleep", return_value=False)
    def test_retries_until_references_section_present(self, _mock_sleep):
        call_count = 0

        class Provider(self._StubProvider):
            def process_document(self, **_kwargs):
                nonlocal call_count
                call_count += 1
                return ReferencesEnforcementTests._BAD if call_count < 2 else ReferencesEnforcementTests._GOOD

        result = _call_llm_with_retry(Provider(), "", False, "sys", "usr", max_retries=3)

        self.assertEqual(call_count, 2)
        self.assertIn("## References", result)

    @patch("summarise.interruptible_sleep", return_value=False)
    def test_raises_after_all_retries_with_missing_references_section(self, _mock_sleep):
        class Provider(self._StubProvider):
            def process_document(self, **_kwargs):
                return ReferencesEnforcementTests._BAD

        with self.assertRaises(ValueError) as ctx:
            _call_llm_with_retry(Provider(), "", False, "sys", "usr", max_retries=2)

        self.assertIn("## References", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
