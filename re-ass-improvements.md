# Re-ass Improvements to Port

Changes in [darrencroton/arxiv-research-assistant](https://github.com/darrencroton/arxiv-research-assistant) from commit [e9cb456](https://github.com/darrencroton/arxiv-research-assistant/commit/e9cb4560b5ee9f57e053ac9611a4b59cbb4eeb4e) to HEAD that need porting here.

---

## 1. OpenAI-compatible API provider

**Commit:** [e9cb456](https://github.com/darrencroton/arxiv-research-assistant/commit/e9cb4560b5ee9f57e053ac9611a4b59cbb4eeb4e)

**The improvement:** Add an `OpenAICompatibleAPI` class — a generic provider for any `/v1/chat/completions` endpoint (LM Studio, llama.cpp, vLLM, LocalAI, etc.). Requires `base_url` and `model` config; optional `api_key_env` for the env var holding the key. No PDF support — text extraction path only.

**Why it was made:** `OllamaAPI` only covers Ollama's native `/api/generate` endpoint. Any other local inference server (the majority, since they expose OpenAI-compatible endpoints) was unsupported. As local model use grows this gap becomes significant.

**Code that needs updating:**
- `providers/api.py` — add `OpenAICompatibleAPI` class (mirrors `OllamaAPI` structure but uses the `openai` SDK pointed at the configured `base_url`)
- `providers/__init__.py` — register `"openai-compatible"` in the API providers dict and import the new class

---

## 2. Runtime readiness check for local providers

**Commit:** [e9cb456](https://github.com/darrencroton/arxiv-research-assistant/commit/e9cb4560b5ee9f57e053ac9611a4b59cbb4eeb4e)

**The improvement:** Add a `validate_runtime_ready()` method to local API providers. For `OpenAICompatibleAPI` it hits `{base_url}/models`, raises a clear `ValueError` if unreachable, and logs a warning if the configured model isn't in the returned list. For `OllamaAPI` an equivalent check against `http://localhost:11434/api/tags` (or `/models`) makes sense.

**Why it was made:** Local inference servers must be running before any paper can be processed. Without this check, failures surface mid-batch after marker-pdf has already done its work, with a confusing connection-refused error rather than a clear startup message.

**Code that needs updating:**
- `providers/base.py` — add a no-op `validate_runtime_ready()` base method
- `providers/api.py` — implement on `OpenAICompatibleAPI` (reachability + model list check); add equivalent to `OllamaAPI`
- `summarise.py` — call `provider.validate_runtime_ready()` in `validate_startup_selection()` or early in `main()` before the processing loop starts

---

## 3. Truncation warning when token limit is hit

**Commits:** [76be207](https://github.com/darrencroton/arxiv-research-assistant/commit/76be207), [dedc086](https://github.com/darrencroton/arxiv-research-assistant/commit/dedc086)

**The improvement:** After receiving a response, check if the provider signals the output was cut at the token limit. For OpenAI-compatible endpoints, `choice.finish_reason == "length"` indicates truncation. For Ollama's `/api/generate`, the equivalent is `done_reason == "length"`. In both cases log a `WARNING` that the summary may be truncated mid-sentence. Also replace bare `logging.warning()` calls in provider files with a module-level `LOGGER = logging.getLogger(__name__)` so warnings can be filtered independently.

**Why it was made:** Local models frequently exhaust their `num_predict`/`max_tokens` budget mid-summary with no visible indication. The result is a structurally broken summary (cut mid-section, missing References, etc.) that passes the empty-check and gets saved without any signal that something went wrong.

**Code that needs updating:**
- `providers/api.py` — add module-level `LOGGER`; in `OpenAICompatibleAPI.process_document()` check `choice.finish_reason == "length"` and log a warning; in `OllamaAPI.process_document()` check `response_data.get("done_reason") == "length"` and log a warning
- `providers/cli.py` — add module-level `LOGGER` (already has one; verify it's used consistently)

---

## 4. Enforce `## References` section in retry loop

**Commit:** [1230a15](https://github.com/darrencroton/arxiv-research-assistant/commit/1230a15)

**The improvement:** Inside `_call_llm_with_retry()`, after the empty-content check, add a structural validation: if the summary contains inline footnote markers (`[^N]`) but no `## References` section heading, raise a `ValueError` and let the retry loop handle it. Two module-level regexes:
```python
_INLINE_FOOTNOTE_RE = re.compile(r"\[\^\d+\]")
_REFERENCES_HEADING_RE = re.compile(r"^## References\s*$", re.MULTILINE)
```

**Why it was made:** Local models frequently produce the inline `[^N]` markers but silently omit the `## References` block at the end (often because they hit the token limit, or because they follow instructions inconsistently). The existing `validate_summary()` only logs a warning — it doesn't trigger a retry. Making this a retryable error means the model gets another chance before a broken summary is saved.

**Code that needs updating:**
- `summarise.py` — add the two regex constants near the top; in `_call_llm_with_retry()`, add the check immediately after the empty-string guard

---

## 5. System prompt rule 13: require `## References` section

**Commit:** [1230a15](https://github.com/darrencroton/arxiv-research-assistant/commit/1230a15)

**The improvement:** Add a 13th rule to the system prompt:
> "ALWAYS include a ## References section at the end listing every footnote definition as [^N]: "exact quote" (Section X.Y, p.Z)"

**Why it was made:** The template already shows a `## References` block, but the rules in the prompt don't explicitly require it. Local models treat the rules list as the authority and frequently skip sections only shown in the template. Making it a named rule significantly increases compliance.

**Code that needs updating:**
- `summarise.py` — `create_system_prompt()`: append rule 13 to the `<rules>` block

---

## 6. Author surname extraction: handle "Last, First" format

**Commit:** [66c2800](https://github.com/darrencroton/arxiv-research-assistant/commit/66c2800)

**The improvement:** In the author-name parsing loop, detect the `"Last, First"` format (presence of a comma within a single author token) and extract the part before the comma as the surname, rather than always taking the first whitespace-delimited word.

```python
# Before
surname = name_parts[0]

# After
if "," in part.strip():
    surname = part.strip().split(",")[0].strip()
else:
    surname = name_parts[0]
```

**Why it was made:** Some PDFs list authors in `"Last, First"` order (common in some journal styles and arXiv metadata fallback paths). The old code would take the wrong token as the surname, producing garbled filenames.

**Code that needs updating:**
- `summarise.py` — `extract_metadata()`: update the inner loop that builds `authors_surnames`

---

## 7. Guard against empty surname from degenerate author strings

**Commit:** [65d48c6](https://github.com/darrencroton/arxiv-research-assistant/commit/65d48c6)

**The improvement:** After splitting a `"Last, First"` name on the comma, the pre-comma part might be empty (e.g., `", R. A."`). Add a fallback to `"Unknown"`:

```python
surname = part.strip().split(",")[0].strip() or "Unknown"
```

**Why it was made:** A degenerate comma-prefixed author string produces an empty string after the split, which then results in a filename starting with ` et al` — a confusing artefact that can cause file-system issues.

**Code that needs updating:**
- `summarise.py` — `extract_metadata()`: same line as improvement 6, add the `or "Unknown"` guard
