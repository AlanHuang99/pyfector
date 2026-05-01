# Changelog

## 0.1.4 - 2026-05-01

- Fixed bootstrap inference when treated outcomes are missing. Overall bootstrap ATT, confidence intervals, and p-values now respect the observed-cell mask.
- Improved matrix-completion performance on rectangular panels by using a Gram-matrix SVD shortcut in the soft-thresholding step.
- Changed the default `n_jobs` to `-1`, using all available CPUs for CV and bootstrap unless specified.
- Made `cv_nobs` create within-unit block masks during cross-validation.
- Changed unsupported `group`, `Z`, and `Q` arguments to raise `NotImplementedError` instead of being ignored.
- Added regression tests for missing treated outcomes, rectangular MC panels, CV block masking, and unsupported argument handling.

## 0.1.3 - 2026-04-09

- Improved bootstrap runtime with relaxed replicate tolerance, warm-started replicates, and EM buffer reuse.
- Kept R comparison tests available for local validation while removing the slow R comparison job from CI.
