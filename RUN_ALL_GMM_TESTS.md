# Run All GMM Test Pack Cases

This package includes two runner helpers:

- `run_all_gmm_tests.py` — Python batch runner with rich summary outputs
- `run_all_gmm_tests.sh` — simple shell wrapper for the common default run

## What the Python runner does

- discovers all CSV files in the test data pack
- runs `gmm_proxy_segmentation.py` once per dataset
- writes one output directory per dataset
- captures `stdout` and `stderr`
- builds batch summaries in JSON, CSV, and Markdown
- applies built-in expected outcomes for the bundled test pack
- supports per-case timeout so one pathological dataset does not block the whole batch

## Recommended command

```bash
python run_all_gmm_tests.py   --script gmm_proxy_segmentation.py   --test-data-dir gmm_test_data_pack   --output-root batch_run_outputs   --fit-on logit   --bootstrap 0   --timeout-seconds 120
```

## Simple shell wrapper

```bash
bash run_all_gmm_tests.sh gmm_proxy_segmentation.py gmm_test_data_pack batch_run_outputs
```

## Output files

Inside the chosen output root:

- `batch_summary.json`
- `batch_summary.csv`
- `batch_summary.md`
- `runs/<dataset_stem>/...`

Each per-case directory may contain:

- `stdout.log`
- `stderr.log`
- `summary.json`
- `segmented_users.csv`
- plot PNGs

## Notes

- `--bootstrap 0` is best for fast regression-style test runs.
- Some datasets are intentionally invalid and are expected to fail.
- One very narrow-range stress dataset is marked with `no_expectation` by default because runtime behavior can vary by environment and package version.
- Use `--ignore-default-expectations` if you want only raw pass/fail reporting.
