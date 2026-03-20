# Provider Mode Plan

## Goal

Replace the current CLI-first, fallback-based provider routing with explicit mode selection:

```bash
python3 summarise.py [mode] [provider] [model]
```

Defaults:

```bash
python3 summarise.py
```

must behave as:

```bash
python3 summarise.py cli claude
```

This is an interface cleanup, not an architecture rewrite.

## Decision

- The user explicitly chooses `cli` or `api`.
- The user explicitly chooses the provider within that mode.
- There is no fallback.
- If the requested mode/provider cannot run, the program fails immediately with a clear error.
- This is a hard cutover. Old one-argument forms such as `python3 summarise.py gemini` are removed rather than kept as compatibility aliases.

## Target Interface

Argument order:

1. `mode`
2. `provider`
3. optional `model`

Examples:

```bash
python3 summarise.py
python3 summarise.py cli claude
python3 summarise.py cli gemini gemini-2.5-flash
python3 summarise.py cli codex gpt-5.4
python3 summarise.py cli copilot
python3 summarise.py api claude claude-sonnet-4-latest
python3 summarise.py api openai gpt-5.2
python3 summarise.py api gemini gemini-2.5-pro
python3 summarise.py api perplexity sonar-pro
python3 summarise.py api ollama llama3.2
```

Expected failure behaviour:

- `cli` + unavailable CLI binary => fail immediately
- `api` + missing API key => fail immediately
- invalid mode => fail immediately
- provider unsupported for selected mode => fail immediately

## Supported Providers By Mode

`cli` mode:

- `claude`
- `gemini`
- `codex`
- `copilot`

`api` mode:

- `claude`
- `gemini`
- `openai`
- `perplexity`
- `ollama`

This preserves the current provider classes while removing ambiguous names such as `claude-api` and `gemini-api`.

## Required Code Changes

### 1. `providers/__init__.py`

Replace the current mixed registry model with two explicit registries:

- `_CLI_PROVIDERS`
- `_API_PROVIDERS`

Change:

```python
create_provider(provider_name, config=None)
```

to:

```python
create_provider(mode, provider_name, config=None)
```

Rules:

- `mode == "cli"` only looks in `_CLI_PROVIDERS`
- `mode == "api"` only looks in `_API_PROVIDERS`
- no PATH-based fallback
- no API-key-based fallback
- errors should name the requested mode, provider, and missing prerequisite

### 2. `summarise.py`

Update argument parsing so startup accepts:

- no args => `cli claude`
- two args => `mode provider`
- three args => `mode provider model`
- one arg or more than three args => fail with usage guidance

Update startup logging so logs clearly show:

- selected mode
- selected provider
- selected model if overridden

The logging should make it impossible to confuse CLI and API runs in `logs/history.log`.

### 3. `start_paper_summariser.sh`

Update the script to pass arguments in the new order and document the new default clearly.

Examples to support:

```bash
./start_paper_summariser.sh
./start_paper_summariser.sh cli gemini
./start_paper_summariser.sh api openai gpt-5.2
```

Remove all comments that mention CLI-first behaviour or fallback.

### 4. `README.md`

Rewrite usage and provider documentation around explicit mode selection.

Must be clear that:

- default is Claude Code CLI
- `cli` and `api` are user-selected modes
- there is no fallback
- failures are intentional and explicit
- old provider names such as `claude-api` are removed

The README should include:

- setup requirements per mode
- valid provider names per mode
- example commands for both `summarise.py` and `start_paper_summariser.sh`
- a short migration note that old one-argument invocations no longer work

Also update any repo-local documentation that still describes CLI-first fallback so internal guidance stays consistent with the shipped interface.

## Implementation Notes

- Reuse existing provider classes. This is a routing change, not a provider rewrite.
- Keep positional arguments rather than introducing a larger CLI framework unless the refactor becomes materially clearer with one.
- Keep the current default model behaviour inside each provider class.
- Do not add compatibility shims unless the rollout decision changes.

## Validation Plan

Minimum manual smoke test set:

1. `python3 summarise.py`
   Expected: starts as `cli claude`
2. `python3 summarise.py cli gemini`
   Expected: selects Gemini CLI only
3. `python3 summarise.py cli codex`
   Expected: selects Codex CLI only
4. `python3 summarise.py api openai`
   Expected: selects OpenAI API only
5. `python3 summarise.py api claude`
   Expected: selects Claude API only

Negative checks:

1. `python3 summarise.py gemini`
   Expected: fail with usage guidance
2. `python3 summarise.py api codex`
   Expected: fail because `codex` is not an API provider
3. `python3 summarise.py cli openai`
   Expected: fail because `openai` is not a CLI provider
4. run a CLI provider with the binary unavailable
   Expected: fail immediately
5. run an API provider with the key unavailable
   Expected: fail immediately

After the refactor:

- run `python3 -m compileall summarise.py providers`
- verify `logs/history.log` shows mode/provider explicitly
- verify `README.md` and script examples match the actual interface exactly

## Delivery Risk

Complexity is low-to-moderate. The main engineering work is small. The real risk is operational:

- existing habits like `./start_paper_summariser.sh gemini` will break
- stale documentation or personal notes could keep old commands alive
- mode/provider naming must stay simple to avoid replacing one ambiguous interface with another

## Recommended Execution Order

1. Refactor `providers/__init__.py`
2. Update argument parsing and logging in `summarise.py`
3. Update `start_paper_summariser.sh`
4. Rewrite `README.md`
5. Run manual smoke tests and negative checks

## Conclusion

This plan is fit for purpose as a direct implementation guide for Task 2. It keeps the change small, removes ambiguous routing, and makes failures explicit instead of silent.
