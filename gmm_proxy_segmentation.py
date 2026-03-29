#!/usr/bin/env python3
"""
GMM-based proxy segmentation for binary-classification prediction scores.

Use case
--------
Given:
  - a probability score column (e.g. LightGBM predict_proba output)
  - a ground-truth binary label column buy in {0,1}

This script fits separate 1D Gaussian Mixture Models to the score distributions in:
  - buy = 1  (actual buyers)
  - buy = 0  (actual non-buyers)

Within each group, it finds score boundaries between mixture components and uses them as
proxy thresholds for:
  - buy = 1: one-time buyer proxy / gray zone / potential repeater proxy
  - buy = 0: non-buyer proxy / gray zone / potential buyer proxy

Important caveat
----------------
Because there is no future label (e.g. repeat_buy_90d or future_buy_90d), these are not
supervised thresholds for the downstream business concepts. They are latent score bands
inside each observed group and should be interpreted as proxies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import os
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



def assign_proxy_segment(scores_prob: Sequence[float], buy_value: int, thresholds_prob: Sequence[float]) -> np.ndarray:
    scores = np.asarray(scores_prob, dtype=float)
    low_thr, high_thr = choose_low_high_thresholds(thresholds_prob)

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



def save_plot(
    group_name: str,
    scores_prob: np.ndarray,
    x_fit: np.ndarray,
    fit_on: str,
    gmm: GaussianMixture,
    thresholds_prob: Sequence[float],
    outpath: Path,
) -> None:
    means, stds, weights = sort_gmm_parameters(gmm)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores_prob, bins=40, density=True)

    grid_prob = np.linspace(float(scores_prob.min()), float(scores_prob.max()), 2000)
    grid_fit = choose_fit_values(grid_prob, fit_on)

    mixture_density_fit = np.column_stack(
        [weights[i] * gaussian_pdf(grid_fit, means[i], stds[i]) for i in range(len(means))]
    )

    if fit_on == "prob":
        component_density_prob = mixture_density_fit
    else:
        # Change of variables: f_p(p) = f_z(logit(p)) * |dz/dp|, dz/dp = 1 / (p (1-p))
        jacobian = 1.0 / np.maximum(grid_prob * (1.0 - grid_prob), 1e-8)
        component_density_prob = mixture_density_fit * jacobian[:, None]

    total_density_prob = component_density_prob.sum(axis=1)

    ax.plot(grid_prob, total_density_prob, linewidth=2.5, label="GMM total density")
    for i in range(component_density_prob.shape[1]):
        ax.plot(grid_prob, component_density_prob[:, i], linewidth=1.5, label=f"component_{i}")

    for j, thr in enumerate(np.sort(np.asarray(thresholds_prob, dtype=float))):
        ax.axvline(float(thr), linestyle="--", linewidth=2, label=f"threshold_{j}={thr:.4f}")

    ax.set_title(f"{group_name}: score distribution and fitted GMM")
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
        "groups": {},
    }

    segmented = work.copy()
    segmented["proxy_segment"] = None

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

        thresholds_prob = summary.thresholds_probability_scale
        low_thr, high_thr = choose_low_high_thresholds(thresholds_prob)

        idx = segmented[args.buy_col] == buy_value
        segmented.loc[idx, "proxy_segment"] = assign_proxy_segment(
            segmented.loc[idx, args.score_col].values,
            buy_value=buy_value,
            thresholds_prob=thresholds_prob,
        )

        segmented.loc[idx, "group_low_threshold"] = low_thr
        segmented.loc[idx, "group_high_threshold"] = high_thr

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
            x_fit=x_fit,
            fit_on=args.fit_on,
            gmm=gmm,
            thresholds_prob=thresholds_prob,
            outpath=plot_path,
        )

        group_payload = asdict(summary)
        group_payload["chosen_low_threshold_probability"] = low_thr
        group_payload["chosen_high_threshold_probability"] = high_thr
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

    # Print a concise report to stdout.
    print("GMM proxy segmentation completed.")
    print(f"Output directory: {output_dir}")
    print(f"Segmented rows: {segmented_path.name}")
    print(f"Summary: {summary_path.name}")
    for group_name, payload in overall_summary["groups"].items():
        if "error" in payload:
            print(f"- {group_name}: {payload['error']}")
            continue
        low_thr = payload.get("chosen_low_threshold_probability")
        high_thr = payload.get("chosen_high_threshold_probability")
        print(
            f"- {group_name}: n_components={payload['n_components']}, "
            f"low_thr={low_thr}, high_thr={high_thr}"
        )

    return 0



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit group-wise GMM thresholds on prediction probabilities and segment proxy user types."
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
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    sys.exit(run(parser.parse_args()))
