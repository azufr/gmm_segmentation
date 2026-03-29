# GMM Proxy Segmentation Plan

## Goal

Use the model's final predicted probability and the current ground truth label `buy` to derive **proxy thresholds** for three score bands inside each observed group:

- `buy = 1`:
  - low-score buyers → **one-time buyer proxy**
  - middle-score buyers → **buyer gray zone**
  - high-score buyers → **potential repeater proxy**
- `buy = 0`:
  - low-score non-buyers → **non-buyer proxy**
  - middle-score non-buyers → **nonbuyer gray zone**
  - high-score non-buyers → **potential buyer proxy**

## Important limitation

This is an **unsupervised proxy segmentation** method.

Because there is **no future label** such as `repeat_buy_90d` or `future_buy_90d`, the thresholds are **not** directly optimized for the downstream concepts “repeater”, “one-time buyer”, or “future buyer”. They are score-based latent bands inferred from the observed score distribution within each current group.

So the outputs should be interpreted as:

- **statistically plausible proxies**, not confirmed behavioral classes
- useful for campaign design, exploration, and prioritization
- something that should later be validated once future-outcome labels become available

---

## Why use GMM here

A Gaussian Mixture Model (GMM) is useful because it can approximate a 1-dimensional score distribution as a mixture of multiple latent subpopulations.

In this problem:

- inside `buy = 1`, the buyer score distribution may contain several latent bands
- inside `buy = 0`, the non-buyer score distribution may also contain several latent bands

The model does not know the business meaning of those bands, but it can often recover clusters like:

- low-confidence observations
- middle ambiguous observations
- high-confidence observations

Those bands become a practical proxy for the user types you want.

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

Your LightGBM output is already a probability between 0 and 1.

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

### Step 1: Prepare the score

- read the score column
- clip the values into `(eps, 1-eps)` to avoid numerical issues
- optionally transform to logit scale

### Step 2: Fit candidate GMMs

Try a small set of candidate models, by default:

- 2 components
- 3 components

The script selects the best model using **BIC**.

Why BIC:

- lower BIC favors a better fit with a complexity penalty
- it is a common practical criterion for choosing mixture component count
- it helps avoid overfitting too many latent bands

### Step 3: Find thresholds from component boundaries

Once the best GMM is fitted, the script builds a dense grid across the fitted score range.

At each grid point, it computes the weighted density contribution from each mixture component and identifies the most likely component. The threshold boundary is defined where the winning component changes.

So if the model selects:

- 2 components → typically 1 threshold
- 3 components → typically 2 thresholds

### Step 4: Convert thresholds back to probability scale

If the model was fitted on logit scale, the threshold values are converted back using:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

This gives thresholds directly on the original predicted probability scale.

### Step 5: Assign proxy segments

For `buy=1`:

- score <= low threshold → `one_time_buyer_proxy`
- score >= high threshold → `potential_repeater_proxy`
- otherwise → `buyer_gray_zone`

For `buy=0`:

- score <= low threshold → `non_buyer_proxy`
- score >= high threshold → `potential_buyer_proxy`
- otherwise → `nonbuyer_gray_zone`

If the model yields only one threshold, the low and high cutoff will be the same, so the segmentation behaves more like a two-band split.

---

## Bootstrap stability check

Thresholds from mixture models can move if the data are noisy.

To check stability, the script optionally bootstraps the thresholds:

1. resample each group with replacement
2. refit the GMM
3. recompute thresholds
4. summarize each threshold with:
   - mean
   - standard deviation
   - median
   - 5th percentile
   - 95th percentile

This helps answer:

- are the thresholds stable?
- does the segmentation change a lot under resampling?

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

- model settings
- number of rows used
- selected component count per group
- BIC per selected model
- component means, standard deviations, and weights
- thresholds on fit scale and probability scale
- chosen low/high cutoffs
- bootstrap summaries
- segment counts

### 3. Plots

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
- `buy` → current ground truth label, 0 or 1

You can override column names with CLI options.

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

### Disable bootstrap for faster runtime

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --bootstrap 0
```

---

## How to interpret the result

### For `buy = 1`

- a low threshold means: buyers below this probability sit in the lowest latent buyer-score band
- a high threshold means: buyers above this probability sit in the highest latent buyer-score band

That highest band is the **best available proxy** for potential repeaters when future repeat labels do not exist.

### For `buy = 0`

- a low threshold means: non-buyers below this probability sit in the lowest latent non-buyer-score band
- a high threshold means: non-buyers above this probability sit in the highest latent non-buyer-score band

That highest band is the **best available proxy** for potential buyers when future conversion labels do not exist.

---

## Pros of this approach

- works without future labels
- easy to operationalize from existing prediction output
- produces interpretable low / middle / high score bands
- can be run separately by current label group
- bootstrap summary gives a practical stability check

---

## Cons and risks

- latent components are not guaranteed to match real behavioral classes
- GMM assumes Gaussian-like component shapes on the chosen fit scale
- thresholds can be unstable if the score distribution is noisy or sample size is small
- if the score is badly calibrated, business interpretation becomes weaker
- 1D segmentation ignores other useful customer features

---

## Recommended next step

Use this script now as a **proxy segmentation baseline**.

Then, when future labels become available, upgrade to a supervised design:

- among `buy=1`, predict repeat purchase
- among `buy=0`, predict future purchase
- tune thresholds with ROC/PR/cost-based methods

That later step will tell you how accurate these proxy bands actually are.

---

## Deliverables included

- `gmm_proxy_segmentation.py` → ready-to-run script
- `GMM_PROXY_SEGMENTATION_PLAN.md` → this implementation and interpretation guide
