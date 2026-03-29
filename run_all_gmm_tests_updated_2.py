#!/usr/bin/env python3
"""
Batch runner for the GMM proxy segmentation test pack.

What it does
------------
- Discovers all CSV files inside a test-data directory
- Runs gmm_proxy_segmentation.py once per file
- Captures stdout/stderr for each case
- Writes a per-case output directory and log files
- Produces JSON/CSV/Markdown summaries across all cases
- Optionally checks expected outcomes for the known bundled test pack

Designed for the synthetic test pack created alongside:
  - gmm_proxy_segmentation.py
  - gmm_test_data_pack/
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_EXPECTATIONS: Dict[str, str] = {
    # expected_success or expected_failure
    "01_happy_path_clear_three_band.csv": "expected_success",
    "02_happy_path_two_band.csv": "expected_success",
    "03_overlap_ambiguous.csv": "expected_success",
    "04_exact_zero_one_and_near_boundaries.csv": "expected_success",
    "05_class_imbalanced.csv": "expected_success",
    "06_missing_values_and_extra_columns.csv": "expected_success",
    "07_rounded_repeated_scores.csv": "expected_success",
    "08_minimum_valid_group_size.csv": "expected_success",
    "09_too_few_rows_in_buy1_group.csv": "expected_failure",
    "10_only_buy1_rows.csv": "expected_success",
    "11_invalid_buy_values.csv": "expected_failure",
    "12_wrong_schema_columns.csv": "expected_failure",
    "13_out_of_range_probabilities.csv": "expected_success",
    "14_constant_repeated_scores.csv": "expected_success",
    "15_narrow_score_range.csv": "no_expectation",
    "16_groupwise_constant_scores.csv": "expected_success",
}


@dataclass
class CaseResult:
    dataset_file: str
    dataset_stem: str
    status: str
    expectation: str
    expectation_result: str
    return_code: int
    duration_seconds: float
    output_dir: str
    stdout_log: str
    stderr_log: str
    summary_file: Optional[str]
    segmented_file: Optional[str]
    current_threshold: Optional[float]
    current_threshold_method: Optional[str]
    current_f1: Optional[float]
    current_precision: Optional[float]
    current_recall: Optional[float]
    low_thr_buy_1: Optional[float]
    high_thr_buy_1: Optional[float]
    low_thr_buy_0: Optional[float]
    high_thr_buy_0: Optional[float]
    notes: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run gmm_proxy_segmentation.py on every CSV in a test-data directory.")
    p.add_argument("--script", required=True, help="Path to gmm_proxy_segmentation.py")
    p.add_argument("--test-data-dir", required=True, help="Directory containing test CSV files")
    p.add_argument("--output-root", required=True, help="Root directory for all run outputs")
    p.add_argument("--python", default=sys.executable, help="Python executable to use")
    p.add_argument("--score-col", default="pred_prob", help="Score column name")
    p.add_argument("--buy-col", default="buy", help="Buy label column name")
    p.add_argument("--fit-on", choices=["prob", "logit"], default="logit")
    p.add_argument("--min-components", type=int, default=2)
    p.add_argument("--max-components", type=int, default=3)
    p.add_argument("--grid-points", type=int, default=5000)
    p.add_argument("--bootstrap", type=int, default=0, help="Bootstrap resamples per case. 0 is fastest for batch runs.")
    p.add_argument("--threshold-policy", choices=["gmm_only", "constrain", "override"], default="constrain")
    p.add_argument("--common-sense-low-ceiling", type=float, default=0.30)
    p.add_argument("--common-sense-high-floor", type=float, default=0.70)
    p.add_argument(
        "--current-threshold-method",
        choices=[
            "none",
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
            "youden_j",
            "precision_constraint",
            "recall_constraint",
        ],
        default="none",
        help="Optional supervised threshold tuning mode for current buy classification.",
    )
    p.add_argument("--current-threshold-min-precision", type=float, default=None)
    p.add_argument("--current-threshold-min-recall", type=float, default=None)
    p.add_argument("--current-threshold-max-candidates", type=int, default=1001)
    p.add_argument("--glob", default="*.csv", help="File glob to discover test datasets")
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Per-case timeout in seconds. Timed-out runs are marked as failures.",
    )
    p.add_argument(
        "--ignore-default-expectations",
        action="store_true",
        help="If set, all datasets are treated as having no predefined expected outcome.",
    )
    return p.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_summary_thresholds(summary_path: Path) -> Dict[str, Optional[float]]:
    defaults = {
        "current_threshold": None,
        "current_threshold_method": None,
        "current_f1": None,
        "current_precision": None,
        "current_recall": None,
        "low_thr_buy_1": None,
        "high_thr_buy_1": None,
        "low_thr_buy_0": None,
        "high_thr_buy_0": None,
    }
    if not summary_path.exists():
        return defaults

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return defaults

    groups = payload.get("groups", {})
    buy1 = groups.get("buy=1", {}) if isinstance(groups, dict) else {}
    buy0 = groups.get("buy=0", {}) if isinstance(groups, dict) else {}
    current = payload.get("current_buy_threshold_search") or {}
    if not isinstance(current, dict):
        current = {}
    return {
        "current_threshold": current.get("selected_threshold"),
        "current_threshold_method": current.get("method"),
        "current_f1": current.get("f1"),
        "current_precision": current.get("precision"),
        "current_recall": current.get("recall"),
        "low_thr_buy_1": buy1.get("chosen_low_threshold_probability") if isinstance(buy1, dict) else None,
        "high_thr_buy_1": buy1.get("chosen_high_threshold_probability") if isinstance(buy1, dict) else None,
        "low_thr_buy_0": buy0.get("chosen_low_threshold_probability") if isinstance(buy0, dict) else None,
        "high_thr_buy_0": buy0.get("chosen_high_threshold_probability") if isinstance(buy0, dict) else None,
    }


def expectation_for_file(filename: str, ignore_defaults: bool) -> str:
    if ignore_defaults:
        return "no_expectation"
    return DEFAULT_EXPECTATIONS.get(filename, "no_expectation")


def evaluate_expectation(expectation: str, return_code: int) -> str:
    success = (return_code == 0)
    if expectation == "expected_success":
        return "matched" if success else "mismatch"
    if expectation == "expected_failure":
        return "matched" if not success else "mismatch"
    return "not_checked"


def build_case_command(args: argparse.Namespace, dataset_path: Path, case_output_dir: Path) -> List[str]:
    cmd = [
        args.python,
        str(Path(args.script)),
        "--input", str(dataset_path),
        "--output-dir", str(case_output_dir),
        "--score-col", args.score_col,
        "--buy-col", args.buy_col,
        "--fit-on", args.fit_on,
        "--min-components", str(args.min_components),
        "--max-components", str(args.max_components),
        "--grid-points", str(args.grid_points),
        "--bootstrap", str(args.bootstrap),
        "--threshold-policy", args.threshold_policy,
        "--common-sense-low-ceiling", str(args.common_sense_low_ceiling),
        "--common-sense-high-floor", str(args.common_sense_high_floor),
        "--current-threshold-method", args.current_threshold_method,
        "--current-threshold-max-candidates", str(args.current_threshold_max_candidates),
    ]
    if args.current_threshold_min_precision is not None:
        cmd.extend(["--current-threshold-min-precision", str(args.current_threshold_min_precision)])
    if args.current_threshold_min_recall is not None:
        cmd.extend(["--current-threshold-min-recall", str(args.current_threshold_min_recall)])
    return cmd


def run_one_case(args: argparse.Namespace, dataset_path: Path, runs_dir: Path) -> CaseResult:
    case_name = dataset_path.stem
    case_output_dir = runs_dir / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = case_output_dir / "stdout.log"
    stderr_log = case_output_dir / "stderr.log"
    summary_file = case_output_dir / "summary.json"
    segmented_file = case_output_dir / "segmented_users.csv"

    cmd = build_case_command(args, dataset_path, case_output_dir)

    start = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        timed_out = False
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        return_code = int(proc.returncode)
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout_text = e.stdout or ""
        stderr_text = e.stderr or ""
        return_code = 124

    end = datetime.now(timezone.utc)
    duration = (end - start).total_seconds()

    stdout_log.write_text(stdout_text, encoding="utf-8")
    stderr_log.write_text(stderr_text, encoding="utf-8")

    status = "success" if return_code == 0 else "failure"
    expectation = expectation_for_file(dataset_path.name, args.ignore_default_expectations)
    expectation_result = evaluate_expectation(expectation, return_code)

    thresholds = load_summary_thresholds(summary_file)
    notes = ""
    if return_code != 0:
        if timed_out:
            notes = f"Timed out after {args.timeout_seconds} seconds"
        else:
            text = (stderr_text or stdout_text or "").strip()
            notes = text.splitlines()[-1] if text else "Run failed"

    return CaseResult(
        dataset_file=dataset_path.name,
        dataset_stem=case_name,
        status=status,
        expectation=expectation,
        expectation_result=expectation_result,
        return_code=int(return_code),
        duration_seconds=float(duration),
        output_dir=str(case_output_dir),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        summary_file=str(summary_file) if summary_file.exists() else None,
        segmented_file=str(segmented_file) if segmented_file.exists() else None,
        current_threshold=thresholds["current_threshold"],
        current_threshold_method=thresholds["current_threshold_method"],
        current_f1=thresholds["current_f1"],
        current_precision=thresholds["current_precision"],
        current_recall=thresholds["current_recall"],
        low_thr_buy_1=thresholds["low_thr_buy_1"],
        high_thr_buy_1=thresholds["high_thr_buy_1"],
        low_thr_buy_0=thresholds["low_thr_buy_0"],
        high_thr_buy_0=thresholds["high_thr_buy_0"],
        notes=notes,
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[CaseResult]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(CaseResult.__annotations__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_markdown_report(
    rows: List[CaseResult],
    args: argparse.Namespace,
    test_data_dir: Path,
    output_root: Path,
) -> str:
    total = len(rows)
    n_success = sum(r.status == "success" for r in rows)
    n_failure = total - n_success
    n_matched = sum(r.expectation_result == "matched" for r in rows)
    n_mismatch = sum(r.expectation_result == "mismatch" for r in rows)

    lines: List[str] = []
    lines.append("# GMM Test Pack Batch Run Report")
    lines.append("")
    lines.append(f"- Run time (UTC): {utc_now_iso()}")
    lines.append(f"- Segmentation script: `{Path(args.script)}`")
    lines.append(f"- Test data directory: `{test_data_dir}`")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Python: `{args.python}`")
    lines.append(f"- Fit on: `{args.fit_on}`")
    lines.append(f"- Bootstrap: `{args.bootstrap}`")
    lines.append(f"- Current threshold method: `{args.current_threshold_method}`")
    lines.append(f"- Per-case timeout: `{args.timeout_seconds}` seconds")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total cases: {total}")
    lines.append(f"- Successes: {n_success}")
    lines.append(f"- Failures: {n_failure}")
    lines.append(f"- Expectation matches: {n_matched}")
    lines.append(f"- Expectation mismatches: {n_mismatch}")
    lines.append("")
    lines.append("## Per-case results")
    lines.append("")
    lines.append("| Dataset | Status | Check | Current thr | Current F1 | buy=1 low | buy=1 high | buy=0 low | buy=0 high | Notes |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            "| {dataset} | {status} | {check} | {ct} | {cf1} | {b1l} | {b1h} | {b0l} | {b0h} | {notes} |".format(
                dataset=r.dataset_file,
                status=r.status,
                check=r.expectation_result,
                ct="" if r.current_threshold is None else f"{r.current_threshold:.6f}",
                cf1="" if r.current_f1 is None else f"{r.current_f1:.6f}",
                b1l="" if r.low_thr_buy_1 is None else f"{r.low_thr_buy_1:.6f}",
                b1h="" if r.high_thr_buy_1 is None else f"{r.high_thr_buy_1:.6f}",
                b0l="" if r.low_thr_buy_0 is None else f"{r.low_thr_buy_0:.6f}",
                b0h="" if r.high_thr_buy_0 is None else f"{r.high_thr_buy_0:.6f}",
                notes=(r.notes or "").replace("|", "/"),
            )
        )

    lines.append("")
    lines.append("## Logs and outputs")
    lines.append("")
    lines.append("Each case has its own output directory under `runs/<dataset_stem>/` containing:")
    lines.append("")
    lines.append("- `stdout.log`")
    lines.append("- `stderr.log`")
    lines.append("- `summary.json` when the run completes far enough to write it")
    lines.append("- `segmented_users.csv` when segmentation succeeds")
    lines.append("- plot PNGs for successful group fits and optional current-threshold search")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    script_path = Path(args.script)
    test_data_dir = Path(args.test_data_dir)
    output_root = Path(args.output_root)
    runs_dir = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2
    if not test_data_dir.exists():
        print(f"ERROR: test-data directory not found: {test_data_dir}", file=sys.stderr)
        return 2

    datasets = sorted(
        p for p in test_data_dir.glob(args.glob)
        if p.is_file()
    )
    if not datasets:
        print(f"ERROR: no datasets found in {test_data_dir} with glob {args.glob}", file=sys.stderr)
        return 2

    results: List[CaseResult] = []
    for dataset_path in datasets:
        result = run_one_case(args, dataset_path, runs_dir)
        results.append(result)
        print(
            f"[{result.status.upper()}] {result.dataset_file} "
            f"(expectation={result.expectation}, check={result.expectation_result}, rc={result.return_code})"
        )

    summary_payload = {
        "run_time_utc": utc_now_iso(),
        "script": str(script_path),
        "test_data_dir": str(test_data_dir),
        "output_root": str(output_root),
        "python": args.python,
        "fit_on": args.fit_on,
        "bootstrap": args.bootstrap,
        "current_threshold_method": args.current_threshold_method,
        "results": [asdict(r) for r in results],
    }

    json_path = output_root / "batch_summary.json"
    csv_path = output_root / "batch_summary.csv"
    md_path = output_root / "batch_summary.md"

    write_json(json_path, summary_payload)
    write_csv(csv_path, results)
    md_path.write_text(build_markdown_report(results, args, test_data_dir, output_root), encoding="utf-8")

    mismatches = sum(r.expectation_result == "mismatch" for r in results)
    print(f"\nWrote:\n- {json_path}\n- {csv_path}\n- {md_path}")
    if mismatches:
        print(f"\nFinished with {mismatches} expectation mismatch(es).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
