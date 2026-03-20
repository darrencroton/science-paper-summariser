# Codebase Modernisation Plan

## Objective

Modernise the science-paper-summariser: fix bugs, eliminate DRY/KISS violations, add CLI tool support as first-class providers (Claude Code, Codex, Gemini CLI, Copilot), restructure the provider layer, and improve overall code quality.

## Scope

- **Included:** All Python source, shell scripts, provider architecture, CLI tool integration, prompt construction, validation, configuration.
- **Excluded:** Prompt engineering / output quality tuning, arXiv auto-fetch (keep modular for later), CI/CD, test framework.

## Context

The codebase has two Python files totalling ~1,260 lines. It was written when LLMs had weaker instruction-following and fewer API features. The user's machine has four AI CLI tools installed (`claude`, `codex`, `gemini`, `copilot`), all supporting non-interactive prompt mode. The user wants **Claude Code CLI as the default provider**, falling back to API when no CLI is available. This changes the project's architecture significantly — from API-only to CLI-first with API fallback.

**Example summary token count:** The provided Harikane et al. example is 18,334 bytes (~4,500 tokens). With the template capping each section at 3 bullets, 8,192 max output tokens provides reasonable headroom, though a modest increase to 12,288 adds safety margin for papers with longer exact quotes.

**Gemini SDK status:** The current `google-generativeai` package is actively maintained and works with all current Gemini models. Migration to `google-genai` is not needed; the only change needed is using the `system_instruction` parameter (already supported).

## Problem Statement

1. **No CLI tool support.** The user's preferred workflow (Claude Code CLI) is not available as a provider. All processing requires API keys and SDK dependencies.
2. **Performance bugs.** marker-pdf models reload per file; LLM provider recreates per file; metadata is extracted twice per successful file.
3. **DRY violations.** Interruptible sleep repeated 4 times; Perplexity reimplements 76 lines of OpenAI-compatible HTTP; shell scripts duplicated.
4. **KISS violations.** Gemini concatenates system prompt into user message instead of using `system_instruction`; validation is 60 lines that never affect control flow; `process_file()` is 130 lines with 4 levels of nesting.
5. **Monolithic provider file.** Adding CLI providers to the existing `llm_providers.py` would push it past 700 lines with two fundamentally different provider types (API vs CLI) mixed together.
6. **Brittle hardcoded model names.** Every provider has a dict mapping model names to context sizes (`llm_providers.py:88-93, 170-178, 251-257, 359-366, 421-425`). These go stale every time a provider ships a new model (which is constantly). The `get_default_model()` methods also hardcode specific model version strings that need manual updating.

## Goals

1. CLI tools (`claude`, `codex`, `gemini`, `copilot`) work as first-class providers via subprocess
2. Auto-detection: if a CLI tool is on PATH, prefer it; fall back to corresponding API provider
3. `claude` (Claude Code CLI) is the default provider
4. Clean provider architecture: `providers/` package with shared base class, separate API and CLI modules
5. Fix all confirmed bugs and DRY/KISS violations from the investigation
6. Output remains unchanged: markdown files following the exact template, deposited in `output/`
7. Keep code modular so arXiv auto-fetch can be added later without restructuring

## Non-Goals

- Structured output / JSON schema intermediate format (adds complexity; current validation-as-safety-net approach is sufficient with modern LLMs)
- arXiv/DOI auto-fetch (deferred; keep code modular for later)
- Migrate Gemini SDK to `google-genai` (current SDK is fine)
- macOS `.command` file changes (not part of repo, personal use)
- Test framework or CI/CD
- Parallel file processing

## Constraints

- Final output must be markdown following the exact template in `project_knowledge/paper-summary-template.md` — this can be ingested directly into Obsidian
- CLI tools vary in how they accept prompts: `claude -p`, `codex exec`, `gemini -p`, `copilot -p`
- CLI tools cannot receive binary PDF bytes; they always need extracted text in the prompt
- Shell argument length limits (~2MB on macOS) are sufficient for paper text
- The service must continue to run as a background `nohup` process via shell scripts
- Provider interface (`process_document()`) must be consistent across CLI and API providers

## Method

- Read all source files line by line
- Traced full processing pipeline from `main()` through providers
- Verified CLI tool availability and flags on the user's machine (`claude`, `codex`, `gemini`, `copilot` all present)
- Examined the ai-orchestrator repo at `github.com/darrencroton/ai-orchestrator` for CLI invocation patterns (subprocess with `-p` flag, stdout capture, signal handling)
- Measured example summary size to assess max_tokens adequacy

## Evidence

| Source | Finding |
|--------|---------|
| `summarise.py:185` | `create_model_dict()` called inside `read_input_file()` — reloads ML models per file |
| `summarise.py:306-307` | `create_llm_provider()` called inside `process_file()` — recreates provider per file |
| `summarise.py:616-617, 640-641` | `extract_metadata()` + `create_base_filename()` called in both `save_summary()` and `move_to_done()` |
| `summarise.py:405-410, 782-784, 788-789, 797-799` | Interruptible sleep pattern repeated 4 times |
| `summarise.py:466` | Dead condition: `line.startswith('  - ')` after lines are stripped at :454 |
| `summarise.py:175-182, 192-194` | Dead commented-out Ollama marker config |
| `summarise.py:369` | `max_tokens=8192` hardcoded |
| `llm_providers.py:312-313` | Gemini concatenates system prompt as string instead of using `system_instruction` |
| `llm_providers.py:185-261` | Perplexity manually builds HTTP requests; API is OpenAI-compatible |
| `llm_providers.py:88-93, 170-178, 251-257, 359-366, 421-425` | Every provider hardcodes model names → context size dicts that go stale with each model release |
| `which claude/codex/gemini/copilot` | All four CLI tools available, all support non-interactive `-p` mode |
| Example summary | 18,334 bytes ≈ 4,500 tokens; 8,192 max_tokens is adequate but tight |

## Options Considered

### Option A: Add CLI providers to existing `llm_providers.py`

- Minimal file changes
- But mixes two fundamentally different patterns (SDK API calls vs subprocess) in one file
- File grows to 700+ lines
- **Rejected:** violates KISS; harder to maintain

### Option B: Restructure into `providers/` package (Recommended)

- `providers/__init__.py` — factory function with auto-detection
- `providers/base.py` — shared `Provider` base class
- `providers/api.py` — all API providers (Claude, OpenAI, Gemini, Perplexity, Ollama)
- `providers/cli.py` — all CLI providers (Claude Code, Codex, Gemini CLI, Copilot)
- Clean separation: API providers share SDK patterns, CLI providers share subprocess patterns
- 4 files, clear responsibilities

### Option C: One file per provider

- 10+ files in `providers/`
- Over-engineered for providers that are each ~40-80 lines
- **Rejected:** violates KISS

## Recommended Approach

**Option B** — restructure into a `providers/` package with four files.

### Provider Architecture

```
providers/
  __init__.py   — create_provider() factory, detect_available_clis(), provider routing
  base.py       — Provider base class (process_document, get_max_context_size, etc.)
  api.py        — ClaudeAPI, OpenAIAPI, GeminiAPI, PerplexityAPI, OllamaProvider
  cli.py        — CLIProvider base, ClaudeCLI, CodexCLI, GeminiCLI, CopilotCLI
```

### Provider Routing Logic

```
User requests "claude" (or default)
  → Is `claude` CLI on PATH?
    → Yes: use ClaudeCLI provider
    → No: is ANTHROPIC_API_KEY set?
      → Yes: use ClaudeAPI provider
      → No: error with clear message

User requests "claude-api"
  → Use ClaudeAPI provider directly (explicit API override)

User requests "gemini"
  → Is `gemini` CLI on PATH?
    → Yes: use GeminiCLI provider
    → No: is GOOGLE_API_KEY set?
      → Yes: use GeminiAPI provider
      → No: error
```

Provider names: `claude`, `codex`, `gemini`, `copilot`, `ollama` auto-detect CLI-first. Explicit API names: `claude-api`, `openai-api`, `gemini-api`, `perplexity-api` bypass auto-detection.

### CLI Provider Implementation Pattern

All CLI providers share the same subprocess pattern:

```python
class CLIProvider(Provider):
    """Base for CLI-based providers."""

    cli_command: str           # e.g. "claude"
    prompt_flag: str           # e.g. "-p"
    extra_flags: list[str]     # e.g. ["--output-format", "text"]

    def setup(self):
        if not shutil.which(self.cli_command):
            raise ValueError(f"{self.cli_command} CLI not found on PATH")

    def supports_direct_pdf(self):
        return False  # CLI tools always get extracted text

    def process_document(self, content, is_pdf, system_prompt, user_prompt, max_tokens=12288):
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        cmd = [self.cli_command, *self.extra_flags, self.prompt_flag, combined_prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"{self.cli_command} failed: {result.stderr[:500]}")
        return result.stdout
```

Each concrete CLI provider only overrides the class attributes and model configuration. Codex is slightly different (uses `exec` subcommand with positional prompt).

### Key Changes to `summarise.py`

1. **Create provider once in `main()`**, pass to `process_file()` — fixes per-file recreation
2. **Lazy-load marker-pdf** — import and cache models only when text extraction is needed
3. **Extract metadata once** — pass title/authors/year from `process_file()` to both `save_summary()` and `move_to_done()`
4. **Extract `interruptible_sleep()` helper** — replaces 4 duplicated patterns
5. **Remove dead code** — commented-out Ollama config, dead `'  - '` validation check
6. **Increase `max_tokens` default to 12,288** — modest increase for safety margin
7. **Simplify `process_file()`** — extract retry loop into helper, reduce nesting

### Key Changes to Providers

1. **Eliminate hardcoded model names.** This is the biggest maintenance burden in the current code. The solution:
   - **Remove `get_default_model()` entirely.** Each provider's SDK/CLI already has its own default. If the user doesn't specify a model, don't pass one — let the SDK/CLI pick. For example, `claude -p "..."` without `--model` uses Claude's current default. `anthropic.Anthropic().messages.create()` without a model requires one, but we can set a single fallback alias like `"claude-sonnet-4-latest"` that the API auto-resolves.
   - **Remove model-to-context-size dicts.** Instead, use a single generous default per provider family (e.g. 200K for Claude, 1M for Gemini, 128K for OpenAI). If the paper is too large, the API will return a clear error on the first retry attempt — no worse than today. The `get_max_context_size()` method becomes a simple constant per provider class, not a dict lookup.
   - **Model override still works.** The user can still pass a specific model via CLI arg (`./start_paper_summariser.sh claude claude-opus-4-6`), which gets forwarded to the provider. But if they don't, it just works with whatever the provider's current default is.
   - This eliminates the need to update the codebase every time a model ships.
2. **Gemini `system_instruction`** — use the SDK parameter instead of string concatenation
3. **Perplexity via OpenAI client** — replace 76 lines of manual HTTP with `openai.OpenAI(base_url="https://api.perplexity.ai")`

## Implementation Phases

### Phase 1: Provider Package Restructure

Create the `providers/` package. Move existing API providers from `llm_providers.py` into `providers/api.py`. Create `providers/base.py` with the shared base class. Create `providers/__init__.py` with the factory function. Archive `llm_providers.py`. Verify existing functionality is unchanged.

**Files created:**
- `providers/__init__.py`
- `providers/base.py`
- `providers/api.py`

**Files modified:**
- `summarise.py` — change import from `llm_providers` to `providers`

**Files archived:**
- `llm_providers.py` → `archive/llm_providers.py`

### Phase 2: CLI Providers

Create `providers/cli.py` with `CLIProvider` base class and concrete implementations for `claude`, `codex`, `gemini`, `copilot`. Add auto-detection logic to `providers/__init__.py`. Update factory to route provider names through CLI-first detection.

**Files created:**
- `providers/cli.py`

**Files modified:**
- `providers/__init__.py` — add CLI routing and auto-detection

### Phase 3: Fix Bugs and DRY/KISS Violations in `summarise.py`

1. Create provider once in `main()`, pass through to `process_file()`
2. Lazy-load marker-pdf imports and cache models at module level
3. Extract metadata once, pass to save/move functions
4. Create `interruptible_sleep()` helper, replace 4 duplicated patterns
5. Remove dead code (commented-out Ollama config, dead validation check)
6. Increase `max_tokens` to 12,288
7. Simplify `process_file()` by extracting retry logic

### Phase 4: API Provider Improvements

1. **Remove hardcoded model names:** Delete all model-to-context-size dicts. Replace `get_default_model()` with SDK/CLI defaults (don't pass a model param unless the user explicitly provides one). Replace `get_max_context_size()` with a single constant per provider family. Remove model lists from README — point users to the provider's own documentation instead.
2. Gemini: use `system_instruction` parameter on `GenerativeModel`
3. Perplexity: replace raw HTTP with `openai.OpenAI(base_url=...)`

### Phase 5: Update Configuration and Documentation

1. Update `README.md` with new provider names and CLI-first behaviour
2. Update `AGENTS.md` to reflect new architecture
3. Update `.env.template` to note API keys are only needed for explicit API providers
4. Update `start_paper_summariser.sh` — default provider is now `claude` (CLI)

## Validation Plan

After each phase:

1. **Phase 1:** Run with `claude-api` provider to verify API path works identically to before
2. **Phase 2:** Run with `claude` provider (CLI) to verify CLI path produces valid summaries. Test with a real PDF. Compare output quality against the Harikane example. Test fallback: rename `claude` binary temporarily, verify it falls back to `claude-api`
3. **Phase 3:** Run with both CLI and API providers. Check logs for: no per-file "Using provider" re-init messages, no marker-pdf model loading for direct-PDF providers, single metadata extraction per file
4. **Phase 4:** Verify providers work without explicit model names (SDK defaults used). Test Gemini specifically (should produce better output with proper `system_instruction`). Test Perplexity via OpenAI client
5. **Phase 5:** Review docs for accuracy

Cross-cutting: The `examples/` directory contains two reference PDFs with their expected summary outputs — use these as ground truth for validation:
- `Harikane et al.pdf` → `Harikane et al - 2026 - A UV-Luminous Galaxy at z = 11 with Surprisingly Weak Star Formation Activity.md`
- `Torralba et al - 2026.pdf` → `Torralba et al - 2026 - The warm outer layer of a little red dot as the source of [Fe ii] and collisiona.md`

Process both PDFs with each provider type and compare output against these references for structural compliance (title, authors, year, sections, footnotes, glossary, tags).

## Risks / Unknowns

- **CLI timeout:** Long papers may cause CLI tools to timeout at the default 300s subprocess limit. Mitigation: make timeout configurable, default higher (600s).
- **CLI prompt length:** Very long papers (~200KB of text) may approach shell argument limits. Mitigation: for prompts exceeding a threshold, write to a temp file and use stdin piping instead. This is unlikely to be hit in practice.
- **CLI output format:** Some CLI tools may include preamble text before the actual summary. Mitigation: the existing `strip_preamble()` function already handles this.
- **Codex exec behaviour:** Codex's `exec` subcommand may behave differently from `claude -p` in terms of tool access and file reading. Need to test.
- **Provider import ordering:** Lazy-loading marker-pdf inside `read_input_file()` requires careful caching to avoid re-importing. Use a module-level sentinel.

## Open Questions

1. **Codex model flag:** Does `codex exec` support a `--model` or `-m` flag for model override? Need to verify at implementation time.
2. **Copilot non-interactive output:** Does `copilot -p` write clean text to stdout, or does it include progress/status information? Need to test.
3. **CLI tool version pinning:** Should we check minimum versions of CLI tools, or just attempt invocation and handle errors?

## Recommended Next Actions

1. Begin Phase 1 (provider package restructure) — this is the foundation everything else builds on
2. Immediately follow with Phase 2 (CLI providers) — this delivers the core new capability
3. Phase 3 (bug fixes) and Phase 4 (API improvements) can be done in either order
4. Phase 5 (docs) at the end once everything is stable

Phases 1-2 are the structural changes. Phases 3-4 are quality improvements. Phase 5 is documentation. Each phase should be a separate commit.

## Completion Status

This plan is complete for its stated scope. All source files were read in full. CLI tool availability was verified on the user's machine. The ai-orchestrator repo was examined for CLI invocation patterns. The example summary was measured for token count. All findings include file:line citations. Implementation phases are ordered by dependency, each with clear deliverables and validation criteria. Open questions are identified for resolution during implementation.
