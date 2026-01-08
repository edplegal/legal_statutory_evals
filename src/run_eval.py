"""Main runner for the landlord-tenant evaluation harness."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from .dataset import DatasetItem, load_dataset
from .models import ModelClient, build_messages, get_model_client_from_env
from .report import write_jsonl, write_report, write_scores_csv
from .scorers import (
    score_legal_correctness,
    score_tn_behavioral_risk_multi,
    score_upl_risk_multi,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "lt_statute_eval.jsonl"
OUTPUTS = ROOT / "outputs"


@contextmanager
def _temp_env(key: str, value: str):
    """Temporarily set an environment variable for the duration of a block."""
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _run_conversation(item: DatasetItem, client: ModelClient) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """Run the model over the item's turns, preserving conversation state.

    Returns:
        final_answer: last assistant message
        history: full message history (user + assistant)
        assistant_turns: list of assistant outputs (one per user turn)
    """
    history: List[Dict[str, str]] = []
    assistant_turns: List[str] = []
    final_answer = ""

    for turn in item.turns:
        history.append({"role": turn["role"], "content": turn["content"]})
        messages = build_messages(history)
        response_text = client.generate(messages)
        history.append({"role": "assistant", "content": response_text})
        assistant_turns.append(response_text)
        final_answer = response_text

    return final_answer, history, assistant_turns


def main() -> None:
    load_dotenv()

    try:
        main_client = get_model_client_from_env()
    except Exception as exc:
        sys.stderr.write(f"Model configuration error: {exc}\n")
        sys.exit(1)

    judge_model_override = os.environ.get("JUDGE_MODEL") or None
    judge_client = (
        main_client
        if not judge_model_override
        else get_model_client_from_env(model_override=judge_model_override)
    )

    try:
        items = load_dataset(DATA_PATH)
    except Exception as exc:
        sys.stderr.write(f"Failed to load dataset: {exc}\n")
        sys.exit(1)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    raw_records: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []

    for item in items:
        # 1) Generate multi-turn conversation (main model)
        try:
            final_answer, conversation, assistant_turns = _run_conversation(item, main_client)
        except Exception as exc:
            sys.stderr.write(f"Error generating for {item.id}: {exc}\n")
            final_answer = f"[generation_failed] {exc}"
            conversation = [{"role": t["role"], "content": t["content"]} for t in item.turns]
            assistant_turns = []

        # 2) Choose what we judge for legal correctness
        # Default: the first assistant response (first substantive answer)
        legal_answer = assistant_turns[0] if assistant_turns else final_answer

        # 3) Legal correctness: run judge deterministically by temporarily setting temp to 0 (or JUDGE_TEMPERATURE)
        judge_temp = os.environ.get("JUDGE_TEMPERATURE", "0")
        with _temp_env("OPENAI_TEMPERATURE", judge_temp):
            legal_score, legal_rationale = score_legal_correctness(legal_answer, item, judge_client)

        # 4) UPL and TN: score across all assistant turns (multi-turn)
        upl_risk, upl_flags = score_upl_risk_multi(assistant_turns)
        tn_scores = score_tn_behavioral_risk_multi(assistant_turns, [t["content"] for t in item.turns])
        tn_violations = tn_scores.get("tn_clause_violations", {}) or {}

        # 5) Write raw record
        raw_records.append(
            {
                **item.to_dict(),
                "model_name": getattr(main_client, "name", "unknown"),
                "conversation": conversation,
                "assistant_turns": assistant_turns,
                "final_answer": final_answer,
                "legal_answer": legal_answer,
                "legal_score": legal_score,
                "legal_rationale": legal_rationale,
            }
        )

        # 6) Write score row
        score_rows.append(
            {
                "id": item.id,
                "scenario": item.scenario,
                "variant": item.variant,
                "model_name": getattr(main_client, "name", "unknown"),
                "legal_correctness": legal_score,
                "upl_risk": upl_risk,
                "tn_risk_level": tn_scores.get("tn_risk_level", "unknown"),
                "tn_clause_hits": tn_scores.get("tn_clause_hits", []),
                "tn_violation_a3": tn_violations.get("TN_2002_A3", False),
                "tn_violation_a4": tn_violations.get("TN_2002_A4", False),
                "tn_violation_a6": tn_violations.get("TN_2002_A6", False),
                "tn_violation_a8": tn_violations.get("TN_2002_A8", False),
                "model_initiated_emotion": tn_scores.get("model_initiated_emotion", False),
                "user_initiated_emotion": tn_scores.get("user_initiated_emotion", False),
                "upl_flags": upl_flags,
                "legal_rationale": legal_rationale,
            }
        )

    write_jsonl(OUTPUTS / "raw_responses.jsonl", raw_records)
    write_scores_csv(OUTPUTS / "scores.csv", score_rows)
    write_report(OUTPUTS / "report.md", score_rows, raw_records)
    print(f"Wrote outputs to {OUTPUTS}")


if __name__ == "__main__":
    main()
