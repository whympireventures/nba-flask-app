# CLAUDE.md — Project Rules & Efficiency Guidelines

## Response Style
- Be terse. Lead with the answer or action, skip preamble.
- No trailing summaries — do not recap what you just did.
- No filler phrases ("Great question!", "Certainly!", "Of course!").
- No unsolicited explanations. If I need context, I'll ask.
- One sentence is better than three when both convey the same meaning.

## Code Behavior
- Only modify code directly related to the request. Do not refactor, clean up, or "improve" surrounding code.
- Do not add docstrings, comments, or type annotations to code you did not change.
- Do not add error handling or validation for scenarios that cannot happen in this codebase.
- Do not create helper utilities or abstractions for one-time operations.
- Do not add logging, print statements, or debug output unless explicitly asked.
- Do not add feature flags, backwards-compatibility shims, or deprecation warnings.
- Prefer editing existing files over creating new ones.
- Never create documentation or README files unless explicitly asked.

## Tool Use
- Read a file before editing it — no blind edits.
- Use dedicated tools (Read, Edit, Grep, Glob) instead of Bash equivalents (cat, grep, find).
- Search directly with Grep/Glob for specific targets. Only use the Explore agent for broad, multi-step searches.
- Run independent tool calls in parallel, not sequentially.

## Planning & Tasks
- Do not propose a plan for simple, clearly scoped tasks — just do it.
- Ask for clarification only when genuinely ambiguous, not as a hedge.

## Project Context
This is a Python/Flask NBA player prop prediction app.

**Key files:**
- `app.py` — Flask routes and app entry point
- `prediction.py` — core prediction logic
- `modeling.py` — model training/loading
- `features.py` — feature engineering
- `api_client.py` — external NBA data API client
- `data_ingest.py` — data pipeline
- `config.py` — configuration constants
- `prizepicks_client.py`, `underdog_client.py`, `parlayplay_client.py` — DFS platform clients
- `injury_client.py`, `live_context.py` — live game context
- `train_models.py` — model training scripts
- `accuracy_test.py`, `historical_backtest.py` — evaluation

**Models (pkl files):** `model_points.pkl`, `model_rebounds.pkl`, `model_assists.pkl`, `model_minutes.pkl`

## What to Avoid
- Do not suggest architectural rewrites unless asked.
- Do not add dependencies without being asked.
- Do not change model files or training logic unless explicitly instructed.
- Do not touch `.venv/` or any installed packages.
