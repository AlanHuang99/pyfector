import numpy as np
import pytest

import pyfector.cv as cv_module
from pyfector.cv import _make_cv_folds


def _reference_make_cv_folds(
    II: np.ndarray,
    D: np.ndarray,
    k: int,
    cv_prop: float,
    cv_nobs: int,
    cv_treat: bool,
    cv_donut: int,
    rng,
) -> list[dict]:
    folds = []
    T, N = II.shape

    if cv_treat:
        ever_treated = np.any(D > 0, axis=0)
        eligible = II.copy()
        eligible[:, ~ever_treated] = 0
    else:
        eligible = II.copy()

    total_eligible = int(np.sum(eligible))
    rm_count = max(1, int(total_eligible * cv_prop))

    eligible_np = np.asarray(eligible)
    elig_rows, elig_cols = np.where(eligible_np > 0)
    n_eligible = len(elig_rows)
    rng_np = np.random.default_rng(int(rng.integers(2**31)) if hasattr(rng, "integers") else 42)

    if n_eligible == 0:
        raise ValueError("No eligible control observations for cross-validation")

    for _ in range(k):
        mask = np.zeros((T, N), dtype=bool)

        if cv_nobs <= 1 or rm_count <= 2:
            chosen = rng_np.choice(n_eligible, size=min(rm_count, n_eligible), replace=False)
            mask[elig_rows[chosen], elig_cols[chosen]] = True
        else:
            while int(mask.sum()) < min(rm_count, n_eligible):
                base = int(rng_np.integers(n_eligible))
                row = int(elig_rows[base])
                col = int(elig_cols[base])
                stop = min(T, row + cv_nobs)
                rows = np.arange(row, stop)
                rows = rows[eligible_np[rows, col] > 0]
                mask[rows, col] = True
                if len(rows) == 0:
                    mask[row, col] = True

            selected = np.flatnonzero(mask.ravel())
            if len(selected) > rm_count:
                drop = rng_np.choice(selected, size=len(selected) - rm_count, replace=False)
                mask.ravel()[drop] = False

        II_after = np.asarray(II).copy()
        II_after[mask] = 0
        bad_rows = np.where(II_after.sum(axis=1) < 1)[0]
        if len(bad_rows) > 0:
            mask[bad_rows, :] = False
            II_after = np.asarray(II).copy()
            II_after[mask] = 0
        orig_col_counts = np.asarray(II).sum(axis=0)
        bad_cols = np.where((orig_col_counts > 0) & (II_after.sum(axis=0) < 1))[0]
        if len(bad_cols) > 0:
            mask[:, bad_cols] = False

        eval_mask = mask.copy()
        if cv_donut > 0:
            for j in range(N):
                treat_times = np.where(np.asarray(D[:, j]) > 0)[0]
                if len(treat_times) > 0:
                    first_treat = treat_times[0]
                    donut_start = max(0, first_treat - cv_donut)
                    donut_end = min(T, first_treat + cv_donut + 1)
                    eval_mask[donut_start:donut_end, j] = False

        folds.append({"mask": np.asarray(mask), "eval": np.asarray(eval_mask)})

    return folds


def _cv_panel() -> tuple[np.ndarray, np.ndarray]:
    T, N = 16, 9
    D = np.zeros((T, N), dtype=bool)
    D[7:, :5] = True
    D[10:, 5] = True

    II = ~D
    II[[1, 4, 8, 12], [0, 2, 5, 7]] = False
    return II, D


@pytest.mark.parametrize("cv_treat, cv_donut", [(True, 0), (True, 2), (False, 0)])
def test_make_cv_folds_counter_preserves_reference_masks(cv_treat, cv_donut):
    II, D = _cv_panel()
    kwargs = dict(
        k=5,
        cv_prop=0.37,
        cv_nobs=3,
        cv_treat=cv_treat,
        cv_donut=cv_donut,
    )

    actual = _make_cv_folds(II, D, rng=np.random.default_rng(987), **kwargs)
    expected = _reference_make_cv_folds(II, D, rng=np.random.default_rng(987), **kwargs)

    for actual_fold, expected_fold in zip(actual, expected, strict=True):
        assert np.array_equal(np.asarray(actual_fold["mask"]), expected_fold["mask"])
        assert np.array_equal(np.asarray(actual_fold["eval"]), expected_fold["eval"])


def test_make_cv_folds_does_not_rescan_mask_sum(monkeypatch):
    T, N = 27, 2_000
    D = np.zeros((T, N), dtype=bool)
    D[15:, :1_000] = True
    II = ~D

    class NoSumMask(np.ndarray):
        def sum(self, *args, **kwargs):
            raise AssertionError("CV fold construction should not call mask.sum()")

    original_zeros = cv_module.np.zeros

    def zeros_without_sum(shape, dtype=float, *args, **kwargs):
        arr = original_zeros(shape, dtype=dtype, *args, **kwargs)
        shape_tuple = tuple(shape) if isinstance(shape, tuple) else (shape,)
        if shape_tuple == (T, N) and np.dtype(dtype) == np.dtype(bool):
            return arr.view(NoSumMask)
        return arr

    monkeypatch.setattr(cv_module.np, "zeros", zeros_without_sum)

    folds = _make_cv_folds(
        II,
        D,
        k=2,
        cv_prop=0.1,
        cv_nobs=3,
        cv_treat=True,
        cv_donut=0,
        rng=np.random.default_rng(123),
    )

    assert len(folds) == 2
    assert all(np.asarray(fold["mask"]).any() for fold in folds)
