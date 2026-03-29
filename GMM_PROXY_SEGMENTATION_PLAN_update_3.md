# GMM Proxy Segmentation Plan

## Goal

Use a model's final predicted probability and the current ground-truth label `buy` to derive **proxy thresholds** for score bands inside each observed group:

- `buy = 1`
  - low-score buyers → `one_time_buyer_proxy`
  - middle-score buyers → `buyer_gray_zone`
  - high-score buyers → `potential_repeater_proxy`
- `buy = 0`
  - low-score non-buyers → `non_buyer_proxy`
  - middle-score non-buyers → `nonbuyer_gray_zone`
  - high-score non-buyers → `potential_buyer_proxy`

This is a **group-wise unsupervised segmentation** workflow built from the current prediction output only.

---

## Important limitation

This method does **not** have future labels such as:

- `repeat_buy_90d`
- `future_buy_90d`

So the thresholds are **not supervised thresholds** for the downstream business concepts “repeater”, “one-time buyer”, or “future buyer”.

Instead, the output should be interpreted as:

- latent score bands inside each current group
- practical proxies for downstream targeting
- something that should later be validated once future-outcome labels become available

---

## What the current script supports

The current `gmm_proxy_segmentation.py` artifact supports:

- CSV and Parquet input
- separate GMM fitting for `buy=1` and `buy=0`
- fitting on either:
  - raw probability with `--fit-on prob`
  - transformed probability with `--fit-on logit`
- trying a range of GMM component counts using BIC
- extracting score boundaries between adjacent latent components
- assigning proxy segments from the discovered boundaries
- optional bootstrap threshold stability summaries
- saving plots, segmented rows, and a JSON summary

The current CLI arguments are:

```bash
--input
--output-dir
--score-col
--buy-col
--fit-on {prob,logit}
--min-components
--max-components
--grid-points
--bootstrap
```

### Current defaults

The current script defaults are:

- `--score-col pred_prob`
- `--buy-col buy`
- `--fit-on logit`
- `--min-components 2`
- `--max-components 3`
- `--grid-points 5000`
- `--bootstrap 200`

Important note: with `min-components=2` and `max-components=3`, BIC may choose **2 components** for a subgroup. In that case, that subgroup naturally yields **one boundary**, which behaves like a **2-band split** rather than a full 3-band split.

If you want to force a 3-band output structure as much as possible, use:

```bash
--min-components 3 --max-components 3
```

---

## Why use GMM here

A Gaussian Mixture Model is useful because it can approximate a 1-dimensional score distribution as a mixture of several latent subpopulations.

In this problem:

- inside `buy = 1`, the buyer score distribution may contain low / middle / high latent bands
- inside `buy = 0`, the non-buyer score distribution may also contain low / middle / high latent bands

The model does not know the business meaning of those bands, but it can often recover structure like:

- low-confidence observations
- middle ambiguous observations
- high-confidence observations

Those latent bands become the practical proxy for the user types you want.

---

## Why fit separately for `buy=1` and `buy=0`

The score distribution of actual buyers and actual non-buyers is usually very different.

If you fit one global GMM over all rows, the model may mostly learn the separation between current buyers and current non-buyers, which is **not** the same as the segmentation you want.

Your use case asks for thresholds **inside each observed outcome group**, so the script fits:

1. one GMM on rows where `buy = 1`
2. another GMM on rows where `buy = 0`

This gives group-specific thresholds.

---

## Why use `logit(probability)` by default

Your model output is already a probability between 0 and 1.

A GMM assumes Gaussian-shaped components, but raw probabilities are bounded and often compressed near 0 or 1. That can make Gaussian fitting on the raw probability scale less natural.

So the script defaults to:

1. clip probability away from exact 0 and 1
2. transform it with:

```text
logit(p) = log(p / (1 - p))
```

This maps the score to the full real line and often makes mixture components more Gaussian-like.

After thresholds are found in logit space, they are converted back to the original probability scale so the final thresholds remain easy to use.

If you prefer, the script can also fit directly on probability with `--fit-on prob`.

---

## Algorithm design

For each group (`buy=1` and `buy=0`), the script does the following.

### Step 1: Read and validate input

- load CSV or Parquet
- confirm `score_col` and `buy_col` exist
- confirm `buy_col` contains only valid binary values after removing missing values
- drop rows with missing score or missing buy label for the actual fitting work

### Step 2: Prepare the score

- clip score values into `(eps, 1-eps)` to avoid numerical issues
- optionally transform to logit scale

### Step 3: Fit candidate GMMs

Try a small set of candidate models from `min_components` to `max_components`.

The script selects the best model using **BIC**.

Why BIC:

- lower BIC favors a better fit with a complexity penalty
- it is a common practical criterion for choosing the number of mixture components
- it helps avoid overfitting too many latent bands

### Step 4: Find boundaries between components

Once the best GMM is fitted, the script builds a dense grid across the fitted score range.

At each grid point, it computes the weighted density contribution from each mixture component and identifies the most likely component. A threshold boundary is defined where the winning component changes.

So if the model selects:

- 2 components → typically 1 threshold
- 3 components → typically 2 thresholds

### Step 5: Convert thresholds back to probability scale

If the model was fitted on logit scale, the threshold values are converted back using:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

This gives thresholds directly on the original predicted-probability scale.

### Step 6: Choose low/high cutoffs for segment assignment

The script turns the discovered probability thresholds into two action cutoffs:

- if there are **2 or more thresholds**, use:
  - lowest threshold as `low`
  - highest threshold as `high`
- if there is **exactly 1 threshold**, use the same value for both `low` and `high`
- if there are **no stable thresholds**, both are `None`

This is why some datasets produce only an effective 2-band split.

### Step 7: Assign proxy segments

For `buy=1`:

- score `<= low_threshold` → `one_time_buyer_proxy`
- score `>= high_threshold` → `potential_repeater_proxy`
- otherwise → `buyer_gray_zone`

For `buy=0`:

- score `<= low_threshold` → `non_buyer_proxy`
- score `>= high_threshold` → `potential_buyer_proxy`
- otherwise → `nonbuyer_gray_zone`

If `low_threshold == high_threshold`, the output behaves like a practical 2-band split with a narrow or empty middle zone around that cutoff.

---

## Bootstrap stability check

Thresholds from mixture models can move if the data are noisy.

To check stability, the script optionally bootstraps the thresholds:

1. resample each group with replacement
2. refit the GMM
3. recompute thresholds
4. summarize each discovered threshold index with:
   - mean
   - standard deviation
   - median
   - 5th percentile
   - 95th percentile
   - successful fits vs attempted fits

This helps answer:

- are the thresholds stable?
- does the segmentation move a lot under resampling?

A narrow bootstrap interval is a good sign.

---

## Script outputs

The script writes the following files.

### 1. `segmented_users.csv`

Contains the original rows plus:

- `proxy_segment`
- `group_low_threshold`
- `group_high_threshold`

### 2. `summary.json`

Contains:

- run settings
- row counts
- group-level fit summaries
- selected component count per group
- BIC per selected model
- component means, standard deviations, and weights
- thresholds on fit scale and probability scale
- chosen low/high cutoffs used for assignment
- bootstrap summaries
- plot file names
- segment counts

### 3. Group plots

One PNG per group:

- histogram of predicted probabilities
- fitted mixture density
- component densities
- vertical threshold lines

These plots help visually inspect whether the thresholds make sense.

---

## Expected input

The script expects at least these columns:

- `pred_prob` → predicted probability from the model
- `buy` → current ground-truth label, 0 or 1

You can override the names with CLI options.

Supported file formats:

- `.csv`
- `.parquet` / `.pq`

---

## Suggested usage

### Basic run

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs
```

### Explicit column names

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --score-col score \
  --buy-col buy_flag
```

### Fit directly on probability instead of logit

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --fit-on prob
```

### Force 3-component search in each subgroup

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --min-components 3 \
  --max-components 3
```

### Disable bootstrap for faster runtime

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --bootstrap 0
```

### Recommended operational run for a 3-band attempt

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --fit-on logit \
  --min-components 3 \
  --max-components 3 \
  --bootstrap 200
```

---

## How to interpret the result

### For `buy = 1`

- a low threshold means buyers below this score sit in the lowest latent buyer-score band
- a high threshold means buyers above this score sit in the highest latent buyer-score band

That highest band is the **best available proxy** for potential repeaters when future repeat labels do not exist.

### For `buy = 0`

- a low threshold means non-buyers below this score sit in the lowest latent non-buyer-score band
- a high threshold means non-buyers above this score sit in the highest latent non-buyer-score band

That highest band is the **best available proxy** for potential buyers when future conversion labels do not exist.

### If low and high are identical

That means the subgroup effectively produced **one stable cutoff** rather than a clean 3-band structure.

Typical reasons:

- BIC selected 2 components
- the score distribution is close to unimodal
- the middle band is weak or not well separated
- the subgroup is too small or too noisy

In that case, forcing `--min-components 3 --max-components 3` may still produce a 3-band output, but the extra band may be weak and should be treated cautiously.

---

## Pros of this approach

- works without future labels
- uses only current prediction output and current ground truth
- easy to operationalize from existing model scores
- produces interpretable low / middle / high score bands when the data support them
- can be run separately by current label group
- bootstrap summary gives a practical stability check
- supports both CSV and Parquet input

---

## Cons and risks

- latent components are not guaranteed to equal real behavioral classes
- if the subgroup distribution is weakly separated, the model may only support 2 effective bands
- thresholds can move with sample size, noise, and component selection
- GMM assumptions may be poor on raw probability scale, especially near 0 or 1
- bootstrap increases runtime
- without future labels, there is no direct validation that “potential repeater” truly repeats

---

## When to use this vs metric-based thresholding

Use this GMM plan when:

- you want **proxy segments inside `buy=1` and `buy=0`**
- you do **not** have future labels for repeat buying or future conversion
- you want exploratory or operational targeting bands

Use metric-based supervised threshold tuning instead when:

- your target is the current `buy` label itself
- you want to optimize a classification metric like F1, precision, recall, or balanced accuracy
- you have the appropriate labels for the decision you actually want to optimize

These are different jobs:

- **GMM** = latent segmentation of score distributions
- **metric tuning** = optimizing a classifier cutoff against known labels

---

## Headless plotting note

If you run this script on a server, terminal-only environment, or container, Matplotlib may fail if it tries to use a GUI backend such as Qt.

Typical fix:

```bash
MPLBACKEND=Agg python gmm_proxy_segmentation.py ...
```

Or set the backend in the script **before** importing `matplotlib.pyplot`:

```python
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
```

---

## Batch test runner note

The companion batch runner `run_all_gmm_tests.py` is intended to call the current script with matching CLI arguments and save per-case outputs, logs, and summary reports.

Before using it with a modified local script, make sure the runner's forwarded flags still match the segmentation script's current `--help` output.

