# GMM Proxy Segmentation Test Data Pack

This pack contains **valid**, **stress**, and **failure-mode** datasets for testing `gmm_proxy_segmentation.py`.

## Included files

### Valid / normal cases
1. `01_happy_path_clear_three_band.csv`
2. `02_happy_path_two_band.csv`
3. `03_overlap_ambiguous.csv`
4. `04_exact_zero_one_and_near_boundaries.csv`
5. `05_class_imbalanced.csv`
6. `06_missing_values_and_extra_columns.csv`
7. `07_rounded_repeated_scores.csv`
8. `08_minimum_valid_group_size.csv`
9. `13_out_of_range_probabilities.csv`
10. `14_constant_repeated_scores.csv`
11. `15_narrow_score_range.csv`
12. `16_groupwise_constant_scores.csv`

### Failure / validation cases
13. `09_too_few_rows_in_buy1_group.csv`
14. `10_only_buy1_rows.csv`
15. `11_invalid_buy_values.csv`
16. `12_wrong_schema_columns.csv`

## Notes
- Synthetic data for pipeline testing.
- Covers normal runs, stress cases, and clear validation failures.
- The pack is broad but not mathematically exhaustive.