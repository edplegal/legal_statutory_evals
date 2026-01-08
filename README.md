Legal advice evaluation harness for Tennessee-style AI chatbot statute risk.

[![Build LaTeX Paper](https://github.com/edplegal/legal_statutory_evals/actions/workflows/latex-paper.yml/badge.svg?branch=latex_ci_dev)](https://github.com/edplegal/legal_statutory_evals/actions/workflows/latex-paper.yml)
[Paper PDF](docs/paper/whitepaper.pdf)

What this does
- Runs 12 landlord-tenant prompts (single and multi-turn) against a chat model.
- Scores along three axes: legal correctness (LLM judge with manual override), UPL risk (heuristic), and Tennessee behavioral risk (heuristic, clause-mapped).
- Saves raw model conversations, a CSV of scores, and a Markdown summary highlighting disagreements.

Quickstart
1) Python 3.11+ and `pip install -r requirements.txt`.
2) Copy `.env.example` to `.env` and fill in model settings.
   - For OpenAI-compatible: set `MODEL_BACKEND=openai_compat`, `OPENAI_BASE_URL` (default https://api.openai.com/v1), `OPENAI_API_KEY`, `OPENAI_MODEL` (e.g., gpt-4o-mini). Optional `JUDGE_MODEL` (defaults to `OPENAI_MODEL`).
   - For Ollama: set `MODEL_BACKEND=ollama`, `OLLAMA_BASE_URL` (default http://localhost:11434), `OLLAMA_MODEL` (e.g., llama3), and optionally `JUDGE_MODEL`.
3) Run the evaluation: `python -m src.run_eval`

Outputs (created in `outputs/`)
- `raw_responses.jsonl`: full prompt and response context for each item.
- `scores.csv`: id, scenario, variant, model name, legal correctness, UPL risk, TN risk level, clause hits, emotion initiation flags.
- `report.md`: label counts, disagreement slices, and a short note on operationalization gaps.

Dataset
- Stored at `data/lt_statute_eval.jsonl`. Schema: `id`, `scenario`, `jurisdiction`, `variant`, `turns` (list of `{role, content}`), `reference_law`, `expected_legal_points`, `manual_labels` (optional).
- Variants per scenario: neutral, user_emotional, multi_turn_continuity (3 user turns with continuity cues).

Scoring
- Axis A (legal correctness): uses `manual_labels.legal_correctness` when present; otherwise runs an LLM judge prompt comparing the answer to `reference_law` and `expected_legal_points`.
- Axis B (UPL risk): deterministic rules for personalized directives, guarantees/threats, and representation/drafting claims; disclaimers lower but do not remove risk.
- Axis C (TN behavioral): deterministic rules mapped to clause IDs (TN_2002_A3 emotional support, TN_2002_A4 companion/relationship, TN_2002_A6 friendship, TN_2002_A8 human simulation). Detects user-initiated vs model-initiated emotional framing.

Statute context
- `docs/tn_statute_excerpt.md` includes the clause excerpts and mapping table used by the heuristics.

White paper
- Source: `docs/paper/whitepaper.tex` (PDF output at `docs/paper/whitepaper.pdf` after build).

Notes
- Keep prompts and outputs local; no training use.
- If no API key is set, the runner will exit with a clear error.
