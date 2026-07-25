"""Task definitions: versioned prompt templates + strict output schemas.

Each task module exposes:
- PROMPTS: dict[version, (system, user_template)]
- Output: pydantic schema the model must satisfy
- build_request(sample_text, version, max_tokens, lang) -> ChatRequest

Prompt version strings are part of the raw-cache key and the results DB
primary key: editing a template REQUIRES bumping its version (v1 -> v2) in
both the module and config/eval.yaml. v1 ships exactly one variant per task
(spec non-goal: no prompt-optimization search).
"""
