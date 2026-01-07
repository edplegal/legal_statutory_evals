"""Main runner for the landlord-tenant evaluation harness."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from .dataset import DatasetItem, load_dataset
from .models import ModelClient, build_messages, get_model_client_from_env
from .report import write_jsonl, write_report, write_scores_csv
from .scorers import score_legal_correctness, score_tn_behavioral_risk, score_upl_risk

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "lt_statute_eval.jsonl"
OUTPUTS = ROOT / "outputs"


def _run_conversation(item: DatasetItem, client: ModelClient) -> Tuple[str, List[Dict[str, str]]]:
    """Run the model over the item's turns, preserving conversation state."""
    history: List[Dict[str, str]] = []
    final_answer = ""
    for turn in item.turns:
        history.append({"role": turn["role"], "content": turn["content"]})
        messages = build_messages(history)
        response_text = client.generate(messages)
        history.append({"role": "assistant", "content": response_text})
        final_answer = response_text
    return final_answer, history


def main() -> None:
    load_dotenv()

    try:
        main_client = get_model_client_from_env()
    except Exception as exc:
        sys.stderr.write(f"Model configuration error: {exc}\n")
        sys.exit(1)

    judge_model_override = os.environ.get("JUDGE_MODEL") or None
    judge_client = main_client if not judge_model_override else get_model_client_from_env(model_override=judge_model_override)

    try:
        items = load_dataset(DATA_PATH)
    except Exception as exc:
        sys.stderr.write(f"Failed to load dataset: {exc}\n")
        sys.exit(1)

    raw_records: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []

    for item in items:
        try:
            final_answer, conversation = _run_conversation(item, main_client)
        except Exception as exc:
            sys.stderr.write(f"Error generating for {item.id}: {exc}\n")
            final_answer = f"[generation_failed] {exc}"
            conversation = item.turns

        legal_score, legal_rationale = score_legal_correctness(final_answer, item, judge_client)
        upl_risk, upl_flags = score_upl_risk(final_answer)
        tn_scores = score_tn_behavioral_risk(final_answer, [turn["content"] for turn in item.turns])
        tn_violations = tn_scores.get("tn_clause_violations", {})

        raw_records.append(
            {
                **item.to_dict(),
                "model_name": getattr(main_client, "name", "unknown"),
                "conversation": conversation,
                "final_answer": final_answer,
                "legal_score": legal_score,
                "legal_rationale": legal_rationale,
            }
        )

        score_rows.append(
            {
                "id": item.id,
                "scenario": item.scenario,
                "variant": item.variant,
                "model_name": getattr(main_client, "name", "unknown"),
                "legal_correctness": legal_score,
                "upl_risk": upl_risk,
                "tn_risk_level": tn_scores["tn_risk_level"],
                "tn_clause_hits": tn_scores["tn_clause_hits"],
                "tn_violation_a3": tn_violations.get("TN_2002_A3", False),
                "tn_violation_a4": tn_violations.get("TN_2002_A4", False),
                "tn_violation_a6": tn_violations.get("TN_2002_A6", False),
                "tn_violation_a8": tn_violations.get("TN_2002_A8", False),
                "model_initiated_emotion": tn_scores["model_initiated_emotion"],
                "user_initiated_emotion": tn_scores["user_initiated_emotion"],
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
