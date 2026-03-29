#!/usr/bin/env python3
"""
GMM-based proxy segmentation plus optional supervised threshold tuning.

Use case
--------
Given:
  - a probability score column (e.g. LightGBM predict_proba output)
  - a ground-truth binary label column buy in {0,1}

This script supports two complementary layers:

1) Proxy segmentation inside each observed group using separate 1D Gaussian Mixture Models
   fit to the score distributions in:
     - buy = 1  (actual buyers)
     - buy = 0  (actual non-buyers)

   Within each group, it finds score boundaries between mixture components and uses them as
   proxy thresholds for:
     - buy = 1: one-time buyer proxy / gray zone / potential repeater proxy
     - buy = 0: non-buyer proxy / gray zone / potential buyer proxy

2) Optional supervised threshold tuning for the *current* buy task using the observed
   ground-truth label. This can optimize a threshold for metrics such as F1, precision,
   recall, balanced accuracy, Youden's J, or constrained objectives.

Important caveat
----------------
Because there is no future label (e.g. repeat_buy_90d or future_buy_90d), the within-group
proxy thresholds are not supervised thresholds for the downstream business concepts. They are
latent score bands inside each observed group and should be interpreted as proxies.

Common-sense threshold policy
-----------------------------
A common business convention is to require:
  - "high" starts no lower than 0.70 probability
  - "low" ends no higher than 0.30 probability

This is a business-rule overlay, not a statistical law. The script supports blending that
rule with the data-driven GMM thresholds via --threshold-policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Force a non-GUI matplotlib backend before importing pyplot so the script works
# in headless environments and on servers without Qt bindings.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


EPS = 1e-6
RANDOM_STATE = 42


@dataclass
class GMMFitSummary:
    group_name: str
    n_samples: int
    fit_scale: str
    n_components: int
    bic: float
    component_means_fit_scale: List[float]
    component_stds_fit_scale: List[float]
    component_weights: List[float]
    thresholds_fit_scale: List[float]
    thresholds_probability_scale: List[float]
    notes: str


@dataclass
class BootstrapSummary:
    threshold_index: int
    mean: float
    std: float
    median: float
    q05: float
    q95: float
    n_success: int
    n_attempted: int


@dataclass
class SupervisedThresholdResult:
    method: str
    selected_threshold: float
    objective_value: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    youden_j: float
    tp: int
    fp: int
    tn: int
    fn: int
    min_precision: Optional[float]
    min_recall: Optional[float]
    threshold_candidates_evaluated: int
    notes: str


def clip_probs(x: Sequence[float], eps: float = EPS) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.clip(arr, eps, 1 - eps)


def logit(p: Sequence[float]) -> np.ndarray:
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


def sigmoid(z: Sequence[float]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-8)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_input_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}. Use CSV or Parquet.")


def validate_columns(df: pd.DataFrame, score_col: str, buy_col: str) -> None:
    missing = [c for c in [score_col, buy_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    unique_buy = set(df[buy_col].dropna().unique().tolist())
    if not unique_buy.issubset({0, 1}):
        raise ValueError(
            f"Column '{buy_col}' must contain only 0/1 values. Found: {sorted(unique_buy)}"
        )


def choose_fit_values(scores_prob: np.ndarray, fit_on: str) -> np.ndarray:
    if fit_on == "prob":
        return clip_probs(scores_prob)
    if fit_on == "logit":
        return logit(scores_prob)
    raise ValueError("fit_on must be either 'prob' or 'logit'")


def inverse_fit_values(x: np.ndarray, fit_on: str) -> np.ndarray:
    if fit_on == "prob":
        return x
    if fit_on == "logit":
        return sigmoid(x)
    raise ValueError("fit_on must be either 'prob' or 'logit'")


def fit_best_gmm_1d(
    x: np.ndarray,
    min_components: int = 2,
    max_components: int = 3,
    random_state: int = RANDOM_STATE,
) -> Tuple[GaussianMixture, float]:
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if len(x) < max(min_components * 5, 20):
        raise ValueError(
            f"Too few rows ({len(x)}) to fit a stable GMM. Need at least 20 samples."
        )

    X = x.reshape(-1, 1)
    best_model: Optional[GaussianMixture] = None
    best_bic = np.inf

    for k in range(min_components, max_components + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=10,
            random_state=random_state,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_model = gmm

    if best_model is None:
        raise RuntimeError("Failed to fit any GMM candidate.")

    return best_model, float(best_bic)


def sort_gmm_parameters(gmm: GaussianMixture) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = gmm.means_.ravel().astype(float)
    covars = gmm.covariances_.reshape(-1).astype(float)
    stds = np.sqrt(np.maximum(covars, 1e-12))
    weights = gmm.weights_.ravel().astype(float)

    order = np.argsort(means)
    return means[order], stds[order], weights[order]


def find_component_boundaries(
    gmm: GaussianMixture,
    x_min: float,
    x_max: float,
    grid_points: int = 5000,
) -> np.ndarray:
    """
    Returns boundaries where the most likely mixture component changes along a dense grid.
    For K components this usually yields K-1 boundaries.
    """
    means, stds, weights = sort_gmm_parameters(gmm)
    grid = np.linspace(float(x_min), float(x_max), int(grid_points))

    weighted = np.column_stack(
        [weights[i] * gaussian_pdf(grid, means[i], stds[i]) for i in range(len(means))]
    )
    winning_component = weighted.argmax(axis=1)
    switch_idx = np.where(winning_component[1:] != winning_component[:-1])[0]
    boundaries = grid[switch_idx]
    return boundaries.astype(float)


def summarize_fit(
    group_name: str,
    x_fit: np.ndarray,
    fit_on: str,
    gmm: GaussianMixture,
    bic: float,
    boundaries_fit_scale: np.ndarray,
) -> GMMFitSummary:
    means, stds, weights = sort_gmm_parameters(gmm)
    boundaries_prob = inverse_fit_values(boundaries_fit_scale, fit_on)

    if len(boundaries_fit_scale) == 0:
        notes = "No stable threshold boundary found from MAP component switches."
    elif len(boundaries_fit_scale) == 1:
        notes = "Single threshold found. Distribution behaves like 2 latent bands."
    else:
        notes = "Multiple thresholds found. Lowest/highest are usually the main action cutoffs."

    return GMMFitSummary(
        group_name=group_name,
        n_samples=int(len(x_fit)),
        fit_scale=fit_on,
        n_components=int(gmm.n_components),
        bic=float(bic),
        component_means_fit_scale=means.tolist(),
        component_stds_fit_scale=stds.tolist(),
        component_weights=weights.tolist(),
        thresholds_fit_scale=boundaries_fit_scale.tolist(),
        thresholds_probability_scale=boundaries_prob.tolist(),
        notes=notes,
    )


def choose_low_high_thresholds(boundaries_prob: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    arr = np.sort(np.asarray(boundaries_prob, dtype=float))
    if len(arr) == 0:
        return None, None
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    return float(arr[0]), float(arr[-1])


def apply_threshold_policy(
    gmm_low: Optional[float],
    gmm_high: Optional[float],
    policy: str,
    low_ceiling: float,
    high_floor: float,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    policy:
      - gmm_only: use pure GMM thresholds
      - constrain: keep GMM thresholds, but enforce low <= low_ceiling and high >= high_floor
      - override: ignore GMM thresholds and use low_ceiling/high_floor directly
    """
    if policy == "gmm_only":
        low, high = gmm_low, gmm_high
        reason = "pure_gmm"
    elif policy == "constrain":
        low = gmm_low
        high = gmm_high

        if low is None:
            low = float(low_ceiling)
        else:
            low = float(min(low, low_ceiling))

        if high is None:
            high = float(high_floor)
        else:
            high = float(max(high, high_floor))

        reason = "gmm_constrained_by_common_sense"
    elif policy == "override":
        low = float(low_ceiling)
        high = float(high_floor)
        reason = "common_sense_override"
    else:
        raise ValueError("threshold policy must be one of: gmm_only, constrain, override")

    if low is not None and high is not None and low > high:
        midpoint = float((low + high) / 2.0)
        low, high = midpoint, midpoint
        reason += "_collapsed_due_to_overlap"

    return low, high, reason


def assign_proxy_segment(
    scores_prob: Sequence[float],
    buy_value: int,
    low_thr: Optional[float],
    high_thr: Optional[float],
) -> np.ndarray:
    scores = np.asarray(scores_prob, dtype=float)

    if buy_value == 1:
        low_label = "one_time_buyer_proxy"
        high_label = "potential_repeater_proxy"
        mid_label = "buyer_gray_zone"
    elif buy_value == 0:
        low_label = "non_buyer_proxy"
        high_label = "potential_buyer_proxy"
        mid_label = "nonbuyer_gray_zone"
    else:
        raise ValueError("buy_value must be 0 or 1")

    labels = np.full(len(scores), mid_label, dtype=object)
    if low_thr is not None:
        labels[scores <= low_thr] = low_label
    if high_thr is not None:
        labels[scores >= high_thr] = high_label
    return labels


def fit_group_model(
    group_df: pd.DataFrame,
    group_name: str,
    score_col: str,
    fit_on: str,
    min_components: int,
    max_components: int,
    grid_points: int,
) -> Tuple[GMMFitSummary, GaussianMixture, np.ndarray, np.ndarray]:
    scores_prob = clip_probs(group_df[score_col].values)
    x_fit = choose_fit_values(scores_prob, fit_on=fit_on)
    gmm, bic = fit_best_gmm_1d(
        x_fit,
        min_components=min_components,
        max_components=max_components,
    )

    boundaries_fit_scale = find_component_boundaries(
        gmm,
        x_min=float(x_fit.min()),
        x_max=float(x_fit.max()),
        grid_points=grid_points,
    )

    summary = summarize_fit(
        group_name=group_name,
        x_fit=x_fit,
        fit_on=fit_on,
        gmm=gmm,
        bic=bic,
        boundaries_fit_scale=boundaries_fit_scale,
    )
    return summary, gmm, scores_prob, x_fit


def bootstrap_thresholds(
    scores_prob: np.ndarray,
    fit_on: str,
    min_components: int,
    max_components: int,
    n_bootstrap: int,
    grid_points: int,
    random_state: int = RANDOM_STATE,
) -> List[BootstrapSummary]:
    if n_bootstrap <= 0:
        return []

    rng = np.random.default_rng(random_state)
    boot_boundaries: List[np.ndarray] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(scores_prob), size=len(scores_prob))
        sample_prob = scores_prob[idx]
        sample_fit = choose_fit_values(sample_prob, fit_on=fit_on)
        try:
            gmm, _ = fit_best_gmm_1d(sample_fit, min_components=min_components, max_components=max_components)
            boundaries_fit = find_component_boundaries(
                gmm,
                x_min=float(sample_fit.min()),
                x_max=float(sample_fit.max()),
                grid_points=grid_points,
            )
            boundaries_prob = np.sort(inverse_fit_values(boundaries_fit, fit_on))
            boot_boundaries.append(boundaries_prob)
        except Exception:
            continue

    if not boot_boundaries:
        return []

    max_len = max(len(x) for x in boot_boundaries)
    summaries: List[BootstrapSummary] = []

    for j in range(max_len):
        vals = [float(arr[j]) for arr in boot_boundaries if len(arr) > j]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        summaries.append(
            BootstrapSummary(
                threshold_index=j,
                mean=float(arr.mean()),
                std=float(arr.std(ddof=0)),
                median=float(np.median(arr)),
                q05=float(np.quantile(arr, 0.05)),
                q95=float(np.quantile(arr, 0.95)),
                n_success=len(vals),
                n_attempted=n_bootstrap,
            )
        )

    return summaries


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    youden_j = recall - (1.0 - specificity)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "youden_j": float(youden_j),
    }


def candidate_thresholds(scores_prob: np.ndarray, max_candidates: int) -> np.ndarray:
    scores = np.sort(np.unique(clip_probs(scores_prob)))
    if len(scores) <= max_candidates:
        return scores

    grid = np.linspace(0.0, 1.0, max_candidates)
    thresholds = np.quantile(scores, grid)
    thresholds = np.unique(np.round(thresholds.astype(float), 12))
    return thresholds


def _method_objective_value(
    method: str,
    metrics: Dict[str, float],
    min_precision: Optional[float],
    min_recall: Optional[float],
) -> Optional[float]:
    if method == "f1":
        return metrics["f1"]
    if method == "precision":
        return metrics["precision"]
    if method == "recall":
        return metrics["recall"]
    if method == "balanced_accuracy":
        return metrics["balanced_accuracy"]
    if method == "youden_j":
        return metrics["youden_j"]
    if method == "precision_constraint":
        target = 0.80 if min_precision is None else float(min_precision)
        if metrics["precision"] < target:
            return None
        return metrics["recall"]
    if method == "recall_constraint":
        target = 0.80 if min_recall is None else float(min_recall)
        if metrics["recall"] < target:
            return None
        return metrics["precision"]
    raise ValueError(
        "current-threshold-method must be one of: none, f1, precision, recall, balanced_accuracy, youden_j, precision_constraint, recall_constraint"
    )


def search_supervised_threshold(
    y_true: np.ndarray,
    scores_prob: np.ndarray,
    method: str,
    max_candidates: int,
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> SupervisedThresholdResult:
    y_true = np.asarray(y_true).astype(int)
    scores_prob = clip_probs(scores_prob)
    thresholds = candidate_thresholds(scores_prob, max_candidates=max_candidates)

    if len(thresholds) == 0:
        raise ValueError("No threshold candidates available.")

    best: Optional[Tuple[float, float, Dict[str, float]]] = None

    for thr in thresholds:
        y_pred = (scores_prob >= thr).astype(int)
        metrics = compute_binary_metrics(y_true, y_pred)
        objective_value = _method_objective_value(
            method=method,
            metrics=metrics,
            min_precision=min_precision,
            min_recall=min_recall,
        )
        if objective_value is None:
            continue

        if best is None:
            best = (float(thr), float(objective_value), metrics)
            continue

        best_thr, best_obj, best_metrics = best
        candidate_key = (
            float(objective_value),
            metrics["f1"],
            metrics["balanced_accuracy"],
            metrics["precision"],
            -float(thr),
        )
        best_key = (
            float(best_obj),
            best_metrics["f1"],
            best_metrics["balanced_accuracy"],
            best_metrics["precision"],
            -float(best_thr),
        )
        if candidate_key > best_key:
            best = (float(thr), float(objective_value), metrics)

    if best is None:
        extra = []
        if method == "precision_constraint":
            extra.append(f"min_precision={min_precision if min_precision is not None else 0.80}")
        if method == "recall_constraint":
            extra.append(f"min_recall={min_recall if min_recall is not None else 0.80}")
        extra_text = ", ".join(extra)
        raise ValueError(f"No threshold satisfied the selected threshold-search objective. {extra_text}".strip())

    thr, objective_value, metrics = best
    notes = "Best threshold found on observed current-buy labels."
    if method == "precision_constraint":
        notes = "Best recall among thresholds meeting the minimum precision constraint."
    elif method == "recall_constraint":
        notes = "Best precision among thresholds meeting the minimum recall constraint."

    return SupervisedThresholdResult(
        method=method,
        selected_threshold=float(thr),
        objective_value=float(objective_value),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        balanced_accuracy=float(metrics["balanced_accuracy"]),
        youden_j=float(metrics["youden_j"]),
        tp=int(metrics["tp"]),
        fp=int(metrics["fp"]),
        tn=int(metrics["tn"]),
        fn=int(metrics["fn"]),
        min_precision=None if min_precision is None else float(min_precision),
        min_recall=None if min_recall is None else float(min_recall),
        threshold_candidates_evaluated=int(len(thresholds)),
        notes=notes,
    )


def save_plot(
    group_name: str,
    scores_prob: np.ndarray,
    fit_on: str,
    gmm: GaussianMixture,
    raw_thresholds_prob: Sequence[float],
    final_low_thr: Optional[float],
    final_high_thr: Optional[float],
    outpath: Path,
) -> None:
    means, stds, weights = sort_gmm_parameters(gmm)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores_prob, bins=40, density=True, alpha=0.45, label="histogram")

    grid_prob = np.linspace(float(scores_prob.min()), float(scores_prob.max()), 2000)
    grid_fit = choose_fit_values(grid_prob, fit_on)

    mixture_density_fit = np.column_stack(
        [weights[i] * gaussian_pdf(grid_fit, means[i], stds[i]) for i in range(len(means))]
    )

    if fit_on == "prob":
        component_density_prob = mixture_density_fit
    else:
        jacobian = 1.0 / np.maximum(grid_prob * (1.0 - grid_prob), 1e-8)
        component_density_prob = mixture_density_fit * jacobian[:, None]

    total_density_prob = component_density_prob.sum(axis=1)

    ax.plot(grid_prob, total_density_prob, linewidth=2.5, label="GMM total density")
    for i in range(component_density_prob.shape[1]):
        ax.plot(grid_prob, component_density_prob[:, i], linewidth=1.5, label=f"component_{i}")

    for j, thr in enumerate(np.sort(np.asarray(raw_thresholds_prob, dtype=float))):
        ax.axvline(float(thr), linestyle=":", linewidth=1.5, label=f"raw_gmm_threshold_{j}={thr:.4f}")

    if final_low_thr is not None:
        ax.axvline(float(final_low_thr), linestyle="--", linewidth=2.0, label=f"final_low={final_low_thr:.4f}")
    if final_high_thr is not None:
        ax.axvline(float(final_high_thr), linestyle="--", linewidth=2.0, label=f"final_high={final_high_thr:.4f}")

    ax.set_title(f"{group_name}: score distribution and fitted GMM")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def save_supervised_threshold_plot(
    y_true: np.ndarray,
    scores_prob: np.ndarray,
    threshold_result: SupervisedThresholdResult,
    outpath: Path,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    scores_prob = clip_probs(scores_prob)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores_prob[y_true == 0], bins=40, alpha=0.45, density=True, label="buy=0")
    ax.hist(scores_prob[y_true == 1], bins=40, alpha=0.45, density=True, label="buy=1")
    ax.axvline(
        threshold_result.selected_threshold,
        linestyle="--",
        linewidth=2.0,
        label=(
            f"selected_threshold={threshold_result.selected_threshold:.4f}\n"
            f"method={threshold_result.method}, f1={threshold_result.f1:.4f}, "
            f"precision={threshold_result.precision:.4f}, recall={threshold_result.recall:.4f}"
        ),
    )
    ax.set_title("Current-buy supervised threshold search")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    df = read_input_table(input_path)
    validate_columns(df, score_col=args.score_col, buy_col=args.buy_col)

    work = df.copy()
    work = work.dropna(subset=[args.score_col, args.buy_col]).copy()
    work[args.score_col] = clip_probs(work[args.score_col].values)
    work[args.buy_col] = work[args.buy_col].astype(int)

    overall_summary: Dict[str, object] = {
        "input_file": str(input_path),
        "n_rows_input": int(len(df)),
        "n_rows_used": int(len(work)),
        "score_col": args.score_col,
        "buy_col": args.buy_col,
        "fit_on": args.fit_on,
        "min_components": args.min_components,
        "max_components": args.max_components,
        "grid_points": args.grid_points,
        "threshold_policy": args.threshold_policy,
        "common_sense_low_ceiling": args.common_sense_low_ceiling,
        "common_sense_high_floor": args.common_sense_high_floor,
        "current_threshold_method": args.current_threshold_method,
        "current_threshold_min_precision": args.current_threshold_min_precision,
        "current_threshold_min_recall": args.current_threshold_min_recall,
        "current_threshold_max_candidates": args.current_threshold_max_candidates,
        "groups": {},
    }

    segmented = work.copy()
    segmented["proxy_segment"] = None

    if args.current_threshold_method != "none":
        current_thr = search_supervised_threshold(
            y_true=work[args.buy_col].values,
            scores_prob=work[args.score_col].values,
            method=args.current_threshold_method,
            max_candidates=args.current_threshold_max_candidates,
            min_precision=args.current_threshold_min_precision,
            min_recall=args.current_threshold_min_recall,
        )
        segmented["current_buy_threshold_method"] = args.current_threshold_method
        segmented["current_buy_threshold"] = float(current_thr.selected_threshold)
        segmented["current_buy_pred"] = (
            segmented[args.score_col].values >= current_thr.selected_threshold
        ).astype(int)

        current_plot_path = output_dir / "plot_current_buy_threshold.png"
        save_supervised_threshold_plot(
            y_true=work[args.buy_col].values,
            scores_prob=work[args.score_col].values,
            threshold_result=current_thr,
            outpath=current_plot_path,
        )

        overall_summary["current_buy_threshold_search"] = asdict(current_thr)
        overall_summary["current_buy_threshold_search"]["plot_file"] = current_plot_path.name
    else:
        overall_summary["current_buy_threshold_search"] = None

    for buy_value, group_name in [(1, "buy=1"), (0, "buy=0")]:
        group_df = work.loc[work[args.buy_col] == buy_value].copy()
        if group_df.empty:
            overall_summary["groups"][group_name] = {"error": "No rows found for this group."}
            continue

        summary, gmm, scores_prob, x_fit = fit_group_model(
            group_df=group_df,
            group_name=group_name,
            score_col=args.score_col,
            fit_on=args.fit_on,
            min_components=args.min_components,
            max_components=args.max_components,
            grid_points=args.grid_points,
        )

        raw_thresholds_prob = summary.thresholds_probability_scale
        raw_low_thr, raw_high_thr = choose_low_high_thresholds(raw_thresholds_prob)
        final_low_thr, final_high_thr, threshold_basis = apply_threshold_policy(
            raw_low_thr,
            raw_high_thr,
            policy=args.threshold_policy,
            low_ceiling=args.common_sense_low_ceiling,
            high_floor=args.common_sense_high_floor,
        )

        idx = segmented[args.buy_col] == buy_value
        segmented.loc[idx, "proxy_segment"] = assign_proxy_segment(
            segmented.loc[idx, args.score_col].values,
            buy_value=buy_value,
            low_thr=final_low_thr,
            high_thr=final_high_thr,
        )

        segmented.loc[idx, "group_raw_low_threshold"] = raw_low_thr
        segmented.loc[idx, "group_raw_high_threshold"] = raw_high_thr
        segmented.loc[idx, "group_low_threshold"] = final_low_thr
        segmented.loc[idx, "group_high_threshold"] = final_high_thr
        segmented.loc[idx, "threshold_policy"] = args.threshold_policy

        boot = bootstrap_thresholds(
            scores_prob=scores_prob,
            fit_on=args.fit_on,
            min_components=args.min_components,
            max_components=args.max_components,
            n_bootstrap=args.bootstrap,
            grid_points=args.grid_points,
        )

        plot_path = output_dir / f"plot_{group_name.replace('=', '_')}.png"
        save_plot(
            group_name=group_name,
            scores_prob=scores_prob,
            fit_on=args.fit_on,
            gmm=gmm,
            raw_thresholds_prob=raw_thresholds_prob,
            final_low_thr=final_low_thr,
            final_high_thr=final_high_thr,
            outpath=plot_path,
        )

        group_payload = asdict(summary)
        group_payload["raw_low_threshold_probability"] = raw_low_thr
        group_payload["raw_high_threshold_probability"] = raw_high_thr
        group_payload["chosen_low_threshold_probability"] = final_low_thr
        group_payload["chosen_high_threshold_probability"] = final_high_thr
        group_payload["threshold_basis"] = threshold_basis
        group_payload["bootstrap_thresholds"] = [asdict(x) for x in boot]
        group_payload["plot_file"] = plot_path.name
        group_payload["segment_counts"] = (
            segmented.loc[idx, "proxy_segment"].value_counts(dropna=False).to_dict()
        )

        overall_summary["groups"][group_name] = group_payload

    segmented_path = output_dir / "segmented_users.csv"
    summary_path = output_dir / "summary.json"
    segmented.to_csv(segmented_path, index=False)
    save_json(summary_path, overall_summary)

    print("GMM proxy segmentation completed.")
    print(f"Output directory: {output_dir}")
    print(f"Segmented rows: {segmented_path.name}")
    print(f"Summary: {summary_path.name}")

    if overall_summary.get("current_buy_threshold_search"):
        current_payload = overall_summary["current_buy_threshold_search"]
        print(
            "- current buy threshold: "
            f"method={current_payload['method']}, "
            f"threshold={current_payload['selected_threshold']}, "
            f"f1={current_payload['f1']}, precision={current_payload['precision']}, recall={current_payload['recall']}"
        )

    for group_name, payload in overall_summary["groups"].items():
        if "error" in payload:
            print(f"- {group_name}: {payload['error']}")
            continue
        raw_low = payload.get("raw_low_threshold_probability")
        raw_high = payload.get("raw_high_threshold_probability")
        low_thr = payload.get("chosen_low_threshold_probability")
        high_thr = payload.get("chosen_high_threshold_probability")
        basis = payload.get("threshold_basis")
        print(
            f"- {group_name}: n_components={payload['n_components']}, "
            f"raw_low={raw_low}, raw_high={raw_high}, "
            f"final_low={low_thr}, final_high={high_thr}, basis={basis}"
        )

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fit group-wise GMM thresholds on prediction probabilities and optionally tune a supervised "
            "current-buy threshold by classification metrics."
        )
    )
    p.add_argument("--input", required=True, help="Path to input CSV or Parquet file.")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    p.add_argument("--score-col", default="pred_prob", help="Probability score column.")
    p.add_argument("--buy-col", default="buy", help="Ground-truth buy column with 0/1 values.")
    p.add_argument(
        "--fit-on",
        choices=["prob", "logit"],
        default="logit",
        help="Fit the GMM on raw probability or on logit(probability).",
    )
    p.add_argument(
        "--min-components",
        type=int,
        default=2,
        help="Minimum number of GMM components to try.",
    )
    p.add_argument(
        "--max-components",
        type=int,
        default=3,
        help="Maximum number of GMM components to try.",
    )
    p.add_argument(
        "--grid-points",
        type=int,
        default=5000,
        help="Number of grid points used to find component boundaries.",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap resamples for threshold stability. Use 0 to disable.",
    )
    p.add_argument(
        "--threshold-policy",
        choices=["gmm_only", "constrain", "override"],
        default="constrain",
        help=(
            "How to combine GMM thresholds with business-rule cutoffs. "
            "'constrain' enforces low <= common_sense_low_ceiling and high >= common_sense_high_floor."
        ),
    )
    p.add_argument(
        "--common-sense-low-ceiling",
        type=float,
        default=0.30,
        help="Business-rule ceiling for the low threshold when threshold-policy is constrain/override.",
    )
    p.add_argument(
        "--common-sense-high-floor",
        type=float,
        default=0.70,
        help="Business-rule floor for the high threshold when threshold-policy is constrain/override.",
    )
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
        help=(
            "Optional supervised threshold search for the current buy-vs-not-buy task. "
            "This uses the observed buy label on the full dataset."
        ),
    )
    p.add_argument(
        "--current-threshold-min-precision",
        type=float,
        default=None,
        help="Minimum precision required when current-threshold-method=precision_constraint.",
    )
    p.add_argument(
        "--current-threshold-min-recall",
        type=float,
        default=None,
        help="Minimum recall required when current-threshold-method=recall_constraint.",
    )
    p.add_argument(
        "--current-threshold-max-candidates",
        type=int,
        default=1001,
        help="Maximum number of candidate thresholds to evaluate for supervised threshold search.",
    )
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    sys.exit(run(parser.parse_args()))
