# R fect vs Python pyfect Comparison

Synthetic panel data comparison between R `fect` and Python `pyfect`.

## Synthetic Comparison

**[View rendered results (HTML)](https://htmlpreview.github.io/?https://github.com/AlanHuang99/pyfect/blob/main/tests/comparison/synthetic_comparison.html)**

### DGP

- 500 units, 12 time periods, 40% staggered treatment
- True ATT = 1.5, 3 covariates (beta = [0.8, -0.5, 0.3])
- 1 interactive factor, two-way fixed effects
- Two scenarios: balanced panel + 5% randomly dropped

### Results summary

| Method | Balanced % diff | 5% Drop % diff |
|--------|----------------|----------------|
| FE     | 0.0%           | 0.0%           |
| IFE    | 5.6%           | 15.2%          |
| MC     | 4.6%           | 4.9%           |

All significance levels match (6/6). Betas match to < 0.002.

## How to render

Requires R with the `fect` package and Python with `pyfect` installed.

```bash
pip install -e .          # install pyfect
quarto render tests/comparison/synthetic_comparison.qmd
```
