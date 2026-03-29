# GMM Proxy Segmentation Plan

## Goal

Use the model's final predicted probability and the current ground truth label `buy` to do **two different jobs**:

1. **Current-buy threshold tuning** using the known label `buy`
   - find a threshold for the main `buy` vs `not buy` decision
   - optionally optimize for `F1`, `precision`, `recall`, `balanced_accuracy`, `Youden's J`, or constrained objectives

2. **Proxy segmentation inside each observed group** using separate GMM fits
   - `buy = 1`:
     - low-score buyers → **one-time buyer proxy**
     - middle-score buyers → **buyer gray zone**
     - high-score buyers → **potential repeater proxy**
   - `buy = 0`:
     - low-score non-buyers → **non-buyer proxy**
     - middle-score non-buyers → **nonbuyer gray zone**
     - high-score non-buyers → **potential buyer proxy**

This gives you a hybrid workflow:

- **supervised where labels exist**
- **unsupervised where future labels do not exist**

---

## Important limitation

The within-group bands are still **unsupervised proxy segments**.

Because there is **no future label** such as `repeat_buy_90d` or `future_buy_90d`, the thresholds inside `buy=1` and `buy=0` are **not** directly optimized for the downstream concepts “repeater”, “one-time buyer”, or “future buyer”. They are score-based latent bands inferred from the observed score distribution within each current group.

So the outputs should be interpreted as:

- **statistically plausible proxies**, not confirmed behavioral classes
- useful for campaign design, exploration, and prioritization
- something that should later be validated once future-outcome labels become available

---

## Architecture of the script

The script now has two layers.

### Layer A: supervised current-buy threshold search

This uses the full dataset and the known label `buy`.

Supported methods:

- `f1`
- `precision`
- `recall`
- `balanced_accuracy`
- `youden_j`
- `precision_constraint`
- `recall_constraint`

This produces:

- one main threshold for `buy` vs `not buy`
- confusion-matrix metrics at that threshold
- a plot showing the class score distributions and selected threshold

### Layer B: GMM proxy segmentation inside each observed group

This uses separate 1D Gaussian Mixture Models for:

- rows where `buy = 1`
- rows where `buy = 0`

Within each group, it finds boundaries between adjacent mixture components and maps them back to the probability scale.

This produces:

- raw GMM thresholds
- final thresholds after applying the threshold policy
- proxy segment labels per row
- group-wise density plots
- optional bootstrap stability summaries

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

## Common-sense threshold policy

A fixed rule like:

- low ≤ 0.30
- high ≥ 0.70

is **not a universal statistical law**. In particular, `0.70` is **not generally “1 standard deviation above the mean”**. Standard deviation depends on the actual distribution, while `0.70` is just a probability value.

Still, `0.30 / 0.70` can be a very reasonable **business convention**:

- below 30% → clearly low propensity
- above 70% → clearly high propensity
- 30% to 70% → gray zone

The script supports 3 policies:

### 1. `gmm_only`

Use the data-driven GMM thresholds exactly as estimated.

### 2. `constrain` (default)

Use the GMM thresholds, but enforce:

- final low threshold ≤ `--common-sense-low-ceiling` (default `0.30`)
- final high threshold ≥ `--common-sense-high-floor` (default `0.70`)

This is the best default when you want a blend of:

- data-driven segmentation
- business common sense

### 3. `override`

Ignore the GMM thresholds and use exactly:

- low = `--common-sense-low-ceiling`
- high = `--common-sense-high-floor`

This is useful when the business rule matters more than the latent score distribution.

---

## When to use supervised threshold tuning instead of GMM

Use supervised threshold tuning when your goal is:

- to **classify current buy vs current non-buy** as well as possible
- to choose a threshold using an explicit prediction metric
- to align the threshold with operational requirements like precision or recall

Use GMM when your goal is:

- to create **proxy low / middle / high bands** inside `buy=1` or inside `buy=0`
- to find natural score regions even though future labels are missing

### Recommendation

Use **both**:

- supervised threshold tuning for the main current-buy cutoff
- GMM for within-group proxy segmentation

This is the design now implemented in the script.

---

## Supervised threshold search methods

### 1. `f1`

Choose the threshold that maximizes F1.

Good when:

- you want a balanced tradeoff between precision and recall
- false positives and false negatives are both important

### 2. `precision`

Choose the threshold that maximizes precision.

Good when:

- false positives are very costly
- you only want highly confident positive predictions

Tradeoff:

- recall may become very low

### 3. `recall`

Choose the threshold that maximizes recall.

Good when:

- missing true buyers is very costly
- coverage matters more than purity

Tradeoff:

- precision may drop sharply

### 4. `balanced_accuracy`

Choose the threshold that maximizes balanced accuracy.

Good when:

- classes are imbalanced
- you care about both sensitivity and specificity

### 5. `youden_j`

Choose the threshold that maximizes:

```text
Youden's J = sensitivity + specificity - 1
```

Good when:

- you want a balanced rule from the ROC perspective
- you want something less sensitive to base rate than raw accuracy

### 6. `precision_constraint`

Among thresholds with precision above a required minimum, choose the one with the best recall.

Good when:

- you need precision to stay above a business floor such as 0.80
- after that, you want as much recall as possible

Relevant option:

- `--current-threshold-min-precision`

### 7. `recall_constraint`

Among thresholds with recall above a required minimum, choose the one with the best precision.

Good when:

- you need recall to stay above a required business floor
- after that, you want the cleanest positives possible

Relevant option:

- `--current-threshold-min-recall`

---

## Other ways to make thresholds more “common sense” without using a fixed value

If you do not want to hard-code `0.30 / 0.70`, there are several reasonable alternatives.

### A. Valley-based thresholding

Keep the threshold at the **density valley** between score bands.

Two practical versions:

- **GMM boundary**: already used in the script
- **Otsu / Multi-Otsu**: choose thresholds that maximize separation between classes in the histogram

This is intuitive when the score distribution shows visible low / middle / high bands.

Good when:

- the histogram has clear structure
- you want a purely data-driven cutoff

Weakness:

- if the histogram is smooth or noisy, the valley can move around a lot

### B. Quantile-based thresholds

Define thresholds from percentiles, for example:

- low = 20th percentile within the subgroup
- high = 80th percentile within the subgroup

This does not assume a Gaussian mixture and guarantees some separation.

Good when:

- you want stable operational segment sizes
- you care about campaign capacity or quota

Weakness:

- quantiles always split the population even if the score distribution has no natural bands

### C. Minimum segment-size rule

Choose a threshold that respects operational scale, for example:

- high segment must contain at least 5% or 10% of the subgroup
- low segment must contain at least 5% or 10% of the subgroup

This is often more practical than a fixed threshold because it avoids:

- a “high” group that is too tiny to act on
- a “low” group that absorbs almost everyone

### D. Stability-based threshold selection

Instead of taking one threshold from one fit, bootstrap the data and prefer thresholds that are:

- stable across resamples
- located in the same neighborhood repeatedly

This is a strong “common sense” rule because unstable cutoffs are hard to trust operationally.

### E. Calibrated probability bands

If the model probabilities are well calibrated, you can define bands using probability meaning rather than geometry:

- low = score range where observed event rate is still low
- high = score range where observed event rate is clearly high

This is more defensible than a fixed `0.70`, but it depends on calibration quality.

Good when:

- current-buy probabilities are reasonably calibrated

Weakness:

- without future labels, this only helps for **current buy risk**, not for repeater vs one-time buyer or future buyer vs non-buyer

### F. Hybrid rule: GMM + business guards

This is usually the most practical choice.

Example:

1. estimate thresholds from GMM
2. require high threshold to land in a plausible range, such as `[0.60, 0.90]`
3. require low threshold to land in a plausible range, such as `[0.10, 0.40]`
4. require each extreme band to contain at least a minimum share of users
5. choose the most stable threshold that satisfies all rules

This avoids both extremes:

- purely arbitrary thresholds
- purely mathematical thresholds that look odd to stakeholders

---

## Recommended policy for this project

Because you do **not** have future labels yet, the best practical approach is:

### For current buy classification

Use one of these supervised threshold methods:

- `f1` as the default general-purpose choice
- `precision_constraint` if false positives are expensive
- `recall_constraint` if missing buyers is expensive

### For proxy segmentation

Use:

- `--threshold-policy constrain`
- `--common-sense-low-ceiling 0.30`
- `--common-sense-high-floor 0.70`
- bootstrap enabled when runtime allows

This gives:

- GMM-based data-driven thresholds
- business-friendly cutoffs
- a stability check

### Best practical hybrid

Use both layers together:

1. tune a **current-buy threshold** with supervised metrics
2. create **within-group proxy bands** with constrained GMM

This gives you a threshold grounded in actual predictive performance plus interpretable proxy segments for CRM or campaign design.

---

## Command-line interface

### Basic proxy segmentation only

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --score-col pred_prob \
  --buy-col buy \
  --fit-on logit \
  --threshold-policy constrain \
  --common-sense-low-ceiling 0.30 \
  --common-sense-high-floor 0.70 \
  --bootstrap 200
```

### Add supervised current-buy threshold search with F1

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --score-col pred_prob \
  --buy-col buy \
  --fit-on logit \
  --threshold-policy constrain \
  --current-threshold-method f1 \
  --bootstrap 200
```

### Current-buy threshold with precision constraint

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --current-threshold-method precision_constraint \
  --current-threshold-min-precision 0.80
```

### Current-buy threshold with recall constraint

```bash
python gmm_proxy_segmentation.py \
  --input prediction_results.csv \
  --output-dir outputs \
  --current-threshold-method recall_constraint \
  --current-threshold-min-recall 0.85
```

---

## Outputs

The script writes:

- `segmented_users.csv`
- `summary.json`
- `plot_buy_1.png`
- `plot_buy_0.png`
- `plot_current_buy_threshold.png` if supervised threshold search is enabled

### `segmented_users.csv`

Includes original columns plus:

- `proxy_segment`
- `group_raw_low_threshold`
- `group_raw_high_threshold`
- `group_low_threshold`
- `group_high_threshold`
- `threshold_policy`
- `current_buy_threshold_method` if enabled
- `current_buy_threshold` if enabled
- `current_buy_pred` if enabled

### `summary.json`

Includes:

- input metadata
- script options used
- supervised threshold-search results for current buy, if enabled
- per-group GMM summaries
- raw and final thresholds
- threshold basis
- bootstrap summaries
- segment counts

---

## Interpretation guide

### Current-buy threshold

This threshold is for the question:

- should this row be classified as a current buyer or not?

It is directly tied to the observed `buy` label and the selected evaluation metric.

### Group-wise proxy thresholds

These thresholds are for the question:

- among actual buyers, who looks low-score vs high-score?
- among actual non-buyers, who looks low-score vs high-score?

These are useful for ranking, prioritization, and strategy design, but they are not confirmed future-behavior labels.

---

## Future improvement path

Once you have future labels, replace the proxy interpretation with true supervised threshold tuning for:

- `repeat_buy_*` inside `buy=1`
- `future_buy_*` inside `buy=0`

At that point, the thresholds can be optimized directly for the business outcomes you actually care about.
