# Batch GMM Test Run Summary

Generated: 2026-03-29T16:19:46+00:00

## Overview

- Total datasets: 16
- Success: 13
- Failure: 3
- Expectation matches: 15
- Expectation mismatches: 0
- Target script: `gmm_proxy_segmentation_update_3.py`
- Forwarded script flags detected from --help: --bootstrap, --buy-col, --fit-on, --grid-points, --max-components, --min-components, --score-col

## Results

| Dataset | Status | Expectation | Match | buy=1 low | buy=1 high | buy=0 low | buy=0 high | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| 01_happy_path_clear_three_band.csv | success | expected_success | matched | 0.307994 | 0.718437 | 0.273362 | 0.737755 |  |
| 02_happy_path_two_band.csv | success | expected_success | matched | 0.148449 | 0.468289 | 0.116893 | 0.422043 |  |
| 03_overlap_ambiguous.csv | success | expected_success | matched | 0.383748 | 0.683006 | 0.282459 | 0.590118 |  |
| 04_exact_zero_one_and_near_boundaries.csv | success | expected_success | matched | 0.000742 | 0.999999 | 1e-06 | 0.99054 |  |
| 05_class_imbalanced.csv | success | expected_success | matched | 0.28195 | 0.742488 | 0.282056 | 0.798962 |  |
| 06_missing_values_and_extra_columns.csv | success | expected_success | matched | 0.286742 | 0.74686 | 0.276447 | 0.524663 |  |
| 07_rounded_repeated_scores.csv | success | expected_success | matched | 0.265502 | 0.708278 | 0.271416 | 0.740089 |  |
| 08_minimum_valid_group_size.csv | success | expected_success | matched | 0.440607 | 0.84725 | 0.102193 | 0.564727 |  |
| 09_too_few_rows_in_buy1_group.csv | failure | expected_failure | matched |  |  |  |  | ValueError: Too few rows (19) to fit a stable GMM. Need at least 20 samples. |
| 10_only_buy1_rows.csv | success | expected_success | matched | 0.279661 | 0.62457 |  |  |  |
| 11_invalid_buy_values.csv | failure | expected_failure | matched |  |  |  |  | ValueError: Column 'buy' must contain only 0/1 values. Found: [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 7.0, 99.0] |
| 12_wrong_schema_columns.csv | failure | expected_failure | matched |  |  |  |  | ValueError: Missing required columns: ['pred_prob', 'buy'] |
| 13_out_of_range_probabilities.csv | success | expected_success | matched | 1e-06 | 0.999999 | 1e-06 | 0.394516 |  |
| 14_constant_repeated_scores.csv | success | expected_success | matched | 0.490199 | 0.844977 | 0.490224 | 0.844918 |  |
| 15_narrow_score_range.csv | success | no_expectation | not_checked | 0.507365 | 0.531306 | 0.474636 | 0.497284 |  |
| 16_groupwise_constant_scores.csv | success | expected_success | matched |  |  |  |  |  |
