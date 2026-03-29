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

This runner is intentionally tolerant of script-version drift:
- it inspects the target script's ``--help`` output
- it forwards only CLI flags that the target script actually supports
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
from typing import Dict, List, Optional, Set


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
    p.add_argument("--min-components", type=int, default=3)
    p.add_argument("--max-components", type=int, default=3)
    p.add_argument("--grid-points", type=int, default=5000)
    p.add_argument("--bootstrap", type=int, default=0, help="Bootstrap resamples per case. 0 is fastest for batch runs.")
    # Legacy / optional passthroughs. These are only forwarded if the target script supports them.
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


def discover_supported_flags(python_bin: str, script_path: str) -> Set[str]:
    """Return the set of long-option names supported by the target script."""
    try:
        proc = subprocess.run(
            [python_bin, script_path, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        return set()

    supported: Set[str] = set()
    for token in help_text.split():
        if token.startswith("--"):
            supported.add(token.rstrip(",;"))
    return supported


def load_summary_thresholds(summary_path: Path) -> Dict[str, Optional[float]]:
    if not summary_path.exists():
        return {
            "low_thr_buy_1": None,
            "high_thr_buy_1": None,
            "low_thr_buy_0": None,
            "high_thr_buy_0": None,
        }

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "low_thr_buy_1": None,
            "high_thr_buy_1": None,
            "low_thr_buy_0": None,
            "high_thr_buy_0": None,
        }

    groups = payload.get("groups", {})
    buy1 = groups.get("buy=1", {}) if isinstance(groups, dict) else {}
    buy0 = groups.get("buy=0", {}) if isinstance(groups, dict) else {}
    return {
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
    success = return_code == 0
    if expectation == "expected_success":
        return "matched" if success else "mismatch"
    if expectation == "expected_failure":
        return "matched" if not success else "mismatch"
    return "not_checked"


def build_case_command(
    args: argparse.Namespace,
    dataset_path: Path,
    case_output_dir: Path,
    supported_flags: Set[str],
) -> List[str]:
    cmd = [
        args.python,
        str(Path(args.script)),
        "--input", str(dataset_path),
        "--output-dir", str(case_output_dir),
    ]

    optional_pairs = [
        ("--score-col", args.score_col),
        ("--buy-col", args.buy_col),
        ("--fit-on", args.fit_on),
        ("--min-components", str(args.min_components)),
        ("--max-components", str(args.max_components)),
        ("--grid-points", str(args.grid_points)),
        ("--bootstrap", str(args.bootstrap)),
    ]

    for flag, value in optional_pairs:
        if flag in supported_flags:
            cmd.extend([flag, value])

    return cmd


def run_one_case(
    args: argparse.Namespace,
    dataset_path: Path,
    runs_dir: Path,
    supported_flags: Set[str],
) -> CaseResult:
    case_name = dataset_path.stem
    case_output_dir = runs_dir / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = case_output_dir / "stdout.log"
    stderr_log = case_output_dir / "stderr.log"
    summary_file = case_output_dir / "summary.json"
    segmented_file = case_output_dir / "segmented_users.csv"

    cmd = build_case_command(args, dataset_path, case_output_dir, supported_flags)

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
            merged = (stderr_text or stdout_text or "").strip().splitlines()
            notes = merged[-1] if merged else "Run failed"
    elif not summary_file.exists():
        notes = "Run succeeded but summary.json was not found"

    return CaseResult(
        dataset_file=dataset_path.name,
        dataset_stem=dataset_path.stem,
        status=status,
        expectation=expectation,
        expectation_result=expectation_result,
        return_code=return_code,
        duration_seconds=duration,
        output_dir=str(case_output_dir),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        summary_file=str(summary_file) if summary_file.exists() else None,
        segmented_file=str(segmented_file) if segmented_file.exists() else None,
        low_thr_buy_1=thresholds["low_thr_buy_1"],
        high_thr_buy_1=thresholds["high_thr_buy_1"],
        low_thr_buy_0=thresholds["low_thr_buy_0"],
        high_thr_buy_0=thresholds["high_thr_buy_0"],
        notes=notes,
    )


def write_csv(path: Path, rows: List[CaseResult]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, rows: List[CaseResult], args: argparse.Namespace, supported_flags: Set[str]) -> None:
    total = len(rows)
    success = sum(1 for r in rows if r.status == "success")
    failure = total - success
    matched = sum(1 for r in rows if r.expectation_result == "matched")
    mismatched = sum(1 for r in rows if r.expectation_result == "mismatch")

    forwarded = sorted(
        x
        for x in [
            "--score-col",
            "--buy-col",
            "--fit-on",
            "--min-components",
            "--max-components",
            "--grid-points",
            "--bootstrap",
        ]
        if x in supported_flags
    )

    lines = [
        "# Batch GMM Test Run Summary",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "## Overview",
        "",
        f"- Total datasets: {total}",
        f"- Success: {success}",
        f"- Failure: {failure}",
        f"- Expectation matches: {matched}",
        f"- Expectation mismatches: {mismatched}",
        f"- Target script: `{args.script}`",
        f"- Forwarded script flags detected from --help: {', '.join(forwarded) if forwarded else '(none beyond required --input/--output-dir)'}",
        "",
        "## Results",
        "",
        "| Dataset | Status | Expectation | Match | buy=1 low | buy=1 high | buy=0 low | buy=0 high | Notes |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]

    for r in rows:
        lines.append(
            f"| {r.dataset_file} | {r.status} | {r.expectation} | {r.expectation_result} | "
            f"{'' if r.low_thr_buy_1 is None else round(r.low_thr_buy_1, 6)} | "
            f"{'' if r.high_thr_buy_1 is None else round(r.high_thr_buy_1, 6)} | "
            f"{'' if r.low_thr_buy_0 is None else round(r.low_thr_buy_0, 6)} | "
            f"{'' if r.high_thr_buy_0 is None else round(r.high_thr_buy_0, 6)} | "
            f"{(r.notes or '').replace('|', '/')} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    test_data_dir = Path(args.test_data_dir)
    if not test_data_dir.exists():
        print(f"ERROR: test-data-dir not found: {test_data_dir}", file=sys.stderr)
        return 2

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2

    supported_flags = discover_supported_flags(args.python, str(script_path))

    output_root = Path(args.output_root)
    runs_dir = output_root / "cases"
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(test_data_dir.rglob(args.glob))
    if not datasets:
        print(f"ERROR: no datasets found under {test_data_dir} with glob {args.glob}", file=sys.stderr)
        return 2

    results: List[CaseResult] = []
    for dataset_path in datasets:
        result = run_one_case(args, dataset_path, runs_dir, supported_flags)
        results.append(result)
        print(f"[{result.status.upper()}] {dataset_path.name} (rc={result.return_code}, {result.duration_seconds:.2f}s)")

    csv_path = output_root / "batch_results.csv"
    json_path = output_root / "batch_results.json"
    md_path = output_root / "batch_results.md"

    write_csv(csv_path, results)
    write_json(
        json_path,
        {
            "generated_at": utc_now_iso(),
            "script": str(script_path),
            "supported_flags": sorted(supported_flags),
            "results": [asdict(r) for r in results],
        },
    )
    write_markdown(md_path, results, args, supported_flags)

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")

    mismatches = [r for r in results if r.expectation_result == "mismatch"]
    if mismatches:
        print(f"\nCompleted with expectation mismatches: {len(mismatches)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
