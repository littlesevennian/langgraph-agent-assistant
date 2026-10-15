"""Intent Classifier evaluation and JSON report persistence."""

import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from src.agent.nodes.intent import classify_intent
from src.agent.utils.errors import AgentError

import argparse

BASE_DIR = Path(__file__).resolve().parent
INTENT_CASES_PATH = BASE_DIR / "eval" / "intent_cases.json"
REPORT_DIR = BASE_DIR / "eval" / "reports"
PROMPT_VERSION = "v3"  # Update manually whenever the intent prompt changes.
MODEL_ID = os.getenv("EVAL_MODEL_ID", "unknown")  # Set to the actual model ID.

INTENTS = ["resume", "project", "learning", "memory_query", "fallback"]
DIFFICULTIES = ["easy", "normal", "boundary"]


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("Intent dataset must be a JSON list")
    return cases


def build_messages(user_input: str) -> list:
    return [HumanMessage(content=user_input)]


def new_stats() -> dict:
    return {
        "total": 0,
        "executed": 0,
        "correct": 0,
        "classification_errors": 0,
        "execution_failures": 0,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def format_rate(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "N/A"


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_report(report: dict[str, Any], report_dir: Path = REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_path = report_dir / f"intent_{PROMPT_VERSION}_{stamp}.json"
    temp_path = output_path.with_suffix(".json.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return output_path


# def evaluate_intent(
#     cases: list[dict],
#     *,
#     dataset_path: Path = INTENT_CASES_PATH,
#     report_dir: Path = REPORT_DIR,
# ) -> dict[str, Any]:
def evaluate_intent(
    cases: list[dict],
    dataset_path: Path = INTENT_CASES_PATH
) -> None:
    total = len(cases)
    correct = 0
    classification_errors: list[dict] = []
    execution_failures: list[dict] = []
    case_results: list[dict] = []
    difficulty_stats = defaultdict(new_stats)
    intent_stats = defaultdict(new_stats)
    started_at = datetime.now(timezone.utc).isoformat()

    for case in cases:
        case_id = case["id"]
        user_input = case["input"]
        expected = case["expected"]
        difficulty = case["difficulty"]
        difficulty_stats[difficulty]["total"] += 1
        intent_stats[expected]["total"] += 1
        record = {
            "id": case_id,
            "input": user_input,
            "difficulty": difficulty,
            "expected": expected,
            "actual": None,
            "confidence": None,
            "reason": None,
            "status": None,
            "error_type": None,
            "error_message": None,
        }
        try:
            result = classify_intent(
                user_input=user_input,
                messages=build_messages(user_input),
            )
        except AgentError as e:
            record.update(
                status="execution_failure",
                error_type=str(e.error_type),
                error_message=str(e),
            )
            execution_failures.append(record)
            difficulty_stats[difficulty]["execution_failures"] += 1
            intent_stats[expected]["execution_failures"] += 1
            case_results.append(record)
            continue

        difficulty_stats[difficulty]["executed"] += 1
        intent_stats[expected]["executed"] += 1
        actual = result.task_type
        record.update(
            actual=actual,
            confidence=result.confidence,
            reason=result.reason,
            status="correct" if actual == expected else "classification_error",
        )
        case_results.append(record)
        if actual == expected:
            correct += 1
            difficulty_stats[difficulty]["correct"] += 1
            intent_stats[expected]["correct"] += 1
        else:
            classification_errors.append(record)
            difficulty_stats[difficulty]["classification_errors"] += 1
            intent_stats[expected]["classification_errors"] += 1

    execution_successes = total - len(execution_failures)
    metrics = {
        "total_cases": total,
        "execution_successes": execution_successes,
        "execution_failures": len(execution_failures),
        "classification_correct": correct,
        "classification_errors": len(classification_errors),
        "execution_success_rate": ratio(execution_successes, total),
        "valid_classification_accuracy": ratio(correct, execution_successes),
        "overall_correct_rate": ratio(correct, total),
    }
    by_difficulty = {key: dict(difficulty_stats[key]) for key in DIFFICULTIES}
    by_intent = {key: dict(intent_stats[key]) for key in INTENTS}
    report = {
        "evaluation_name": "intent_classifier",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model_id": MODEL_ID,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256(dataset_path),
        "metrics": metrics,
        "by_difficulty": by_difficulty,
        "by_intent": by_intent,
        "results": case_results,
    }

    print("\n========== Intent Evaluation ==========")
    print(f"Total Cases                : {total}")
    print(f"Execution Successes        : {execution_successes}")
    print(f"Execution Failures         : {len(execution_failures)}")
    print(f"Classification Correct     : {correct}")
    print(f"Classification Errors      : {len(classification_errors)}")
    print(f"Execution Success Rate     : {format_rate(metrics['execution_success_rate'])}")
    print(f"Valid Classification Acc.  : {format_rate(metrics['valid_classification_accuracy'])}")
    print(f"Overall Correct Rate       : {format_rate(metrics['overall_correct_rate'])}")

    for heading, keys, stats_map in (
        ("Accuracy by Difficulty", DIFFICULTIES, by_difficulty),
        ("Accuracy by Intent", INTENTS, by_intent),
    ):
        print(f"\n========== {heading} ==========")
        for key in keys:
            stats = stats_map[key]
            accuracy = ratio(stats["correct"], stats["total"])
            print(
                f"{key:<12}: correct={stats['correct']}/{stats['total']} "
                f"accuracy={format_rate(accuracy)} "
                f"execution_failures={stats['execution_failures']}"
            )

    if classification_errors:
        print("\n========== Classification Errors ==========")
        for item in classification_errors:
            print("-" * 60)
            for key in ("id", "difficulty", "input", "expected", "actual", "confidence", "reason"):
                print(f"{key:<12}: {item[key]}")

    if execution_failures:
        print("\n========== Execution Failures ==========")
        for item in execution_failures:
            print("-" * 60)
            for key in ("id", "difficulty", "input", "expected", "error_type", "error_message"):
                print(f"{key:<12}: {item[key]}")

    saved_path = save_report(report, REPORT_DIR)
    print(f"\nJSON Report Saved          : {saved_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Intent Classifier Evaluation Runner"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=INTENT_CASES_PATH,
        help="Path to the evaluation dataset JSON file"
    )

    args = parser.parse_args()

    dataset_path = args.dataset.resolve()

    if not dataset_path.is_file():
        parser.error(f"Dataset file not found: {dataset_path}")

    cases = load_cases(dataset_path)
    # evaluate_intent(cases)
    evaluate_intent(cases, dataset_path=dataset_path)


if __name__ == "__main__":
    main()
