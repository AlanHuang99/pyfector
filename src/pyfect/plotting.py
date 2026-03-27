"""
Plotting functions for pyfect results.

Produces matplotlib/plotly figures equivalent to R fect's plot types:
- gap: dynamic treatment effects with CI bands
- status: treatment status heatmap
- factors: latent factor time series
- counterfactual: actual vs counterfactual for selected units
- equiv: equivalence test visualization
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def plot(
    result,
    kind: Literal["gap", "status", "factors", "counterfactual", "equiv", "calendar"] = "gap",
    units: list | None = None,
    show_ci: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    ax=None,
    **kwargs,
):
    """Plot fect results.

    Parameters
    ----------
    result : FectResult
        Output from ``pyfect.fect()``.
    kind : str
        Plot type.
    units : list, optional
        Unit IDs for counterfactual plot.
    show_ci : bool
        Show confidence intervals (requires ``se=True`` in estimation).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if kind == "gap":
        _plot_gap(result, ax, show_ci, **kwargs)
    elif kind == "status":
        _plot_status(result, ax, **kwargs)
    elif kind == "factors":
        _plot_factors(result, ax, **kwargs)
    elif kind == "counterfactual":
        _plot_counterfactual(result, ax, units, **kwargs)
    elif kind == "equiv":
        _plot_equiv(result, ax, **kwargs)
    elif kind == "calendar":
        _plot_calendar(result, ax, show_ci, **kwargs)
    else:
        raise ValueError(f"Unknown plot kind: {kind}")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def _plot_gap(result, ax, show_ci, count_bar: bool = True, **kwargs):
    """Dynamic treatment effects (gap) plot."""
    time_on = result.time_on
    att_on = result.att_on

    if time_on is None or att_on is None:
        ax.text(0.5, 0.5, "No dynamic effects available", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Main line
    ax.plot(time_on, att_on, "o-", color="#2166ac", markersize=4, linewidth=1.5,
            label="ATT", zorder=3)

    # CI bands
    if show_ci and result.inference is not None:
        inf = result.inference
        ax.fill_between(
            time_on, inf.att_on_ci_lower, inf.att_on_ci_upper,
            alpha=0.2, color="#2166ac", label="95% CI",
        )

    # Zero line
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    # Treatment onset line
    ax.axvline(0, color="#b2182b", linewidth=1, linestyle=":", alpha=0.7,
               label="Treatment onset", zorder=2)

    # Count bars at bottom
    if count_bar and result.count_on is not None:
        ax2 = ax.twinx()
        ax2.bar(time_on, result.count_on, alpha=0.15, color="gray", width=0.8)
        ax2.set_ylabel("Count", color="gray", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
        max_count = max(result.count_on) if len(result.count_on) > 0 else 1
        ax2.set_ylim(0, max_count * 4)

    ax.set_xlabel("Relative time to treatment")
    ax.set_ylabel("Treatment effect")
    ax.legend(loc="best", fontsize=9)


def _plot_status(result, ax, **kwargs):
    """Treatment status heatmap."""
    import matplotlib.colors as mcolors

    panel = result.panel
    if panel is None:
        return

    status = panel.obs_status

    # Custom colormap: 1=treated(red), 2=control(blue), 3=missing(gray), 4=removed(white)
    cmap = mcolors.ListedColormap(["#b2182b", "#2166ac", "#d9d9d9", "#ffffff"])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Sort units: controls first, then treated by treatment timing
    order = np.argsort(panel.unit_type)

    im = ax.imshow(
        status[:, order].T, aspect="auto", cmap=cmap, norm=norm,
        interpolation="nearest",
    )

    ax.set_xlabel("Time period")
    ax.set_ylabel("Unit")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#b2182b", label="Treated"),
        Patch(facecolor="#2166ac", label="Control"),
        Patch(facecolor="#d9d9d9", label="Missing"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)


def _plot_factors(result, ax, **kwargs):
    """Plot estimated latent factors over time."""
    if result.factors is None:
        ax.text(0.5, 0.5, "No factors estimated", ha="center", va="center",
                transform=ax.transAxes)
        return

    T, r = result.factors.shape
    time_ids = result.panel.time_ids if result.panel is not None else np.arange(T)

    for k in range(r):
        ax.plot(time_ids, result.factors[:, k], label=f"Factor {k+1}", linewidth=1.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Factor value")
    ax.legend(fontsize=9)


def _plot_counterfactual(result, ax, units=None, **kwargs):
    """Actual vs counterfactual outcomes for selected units."""
    panel = result.panel
    if panel is None or result.Y_ct is None:
        return

    if units is None:
        # Pick first 3 treated units
        treated_idx = np.where(panel.unit_type == 2)[0][:3]
    else:
        treated_idx = [np.where(panel.unit_ids == u)[0][0] for u in units
                       if u in panel.unit_ids]

    time_ids = panel.time_ids
    colors = ["#2166ac", "#b2182b", "#4daf4a", "#984ea3", "#ff7f00"]

    for i, j in enumerate(treated_idx):
        obs_mask = panel.I[:, j] > 0
        color = colors[i % len(colors)]
        uid = panel.unit_ids[j]

        ax.plot(time_ids[obs_mask], panel.Y[obs_mask, j], "o-",
                color=color, markersize=3, linewidth=1,
                label=f"Unit {uid} (actual)")
        ax.plot(time_ids, result.Y_ct[:, j], "--",
                color=color, linewidth=1, alpha=0.7,
                label=f"Unit {uid} (counterfactual)")

    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.legend(fontsize=8)


def _plot_equiv(result, ax, threshold: float = 0.36, **kwargs):
    """Equivalence test visualization."""
    if result.att_on is None or result.inference is None:
        ax.text(0.5, 0.5, "Inference required", ha="center", va="center",
                transform=ax.transAxes)
        return

    inf = result.inference
    # Only pre-treatment periods
    pre_mask = result.time_on < 0
    if not np.any(pre_mask):
        return

    t_pre = result.time_on[pre_mask]
    att_pre = result.att_on[pre_mask]
    ci_lo = inf.att_on_ci_lower[pre_mask]
    ci_hi = inf.att_on_ci_upper[pre_mask]

    ax.errorbar(t_pre, att_pre, yerr=[att_pre - ci_lo, ci_hi - att_pre],
                fmt="o", color="#2166ac", capsize=3, markersize=4)

    # Equivalence bounds
    ax.axhspan(-threshold, threshold, alpha=0.1, color="green",
               label=f"Equiv. bounds (+-{threshold})")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Relative time (pre-treatment)")
    ax.set_ylabel("ATT")
    ax.legend(fontsize=9)


def _plot_calendar(result, ax, show_ci, **kwargs):
    """Calendar-time ATT plot."""
    if result.eff is None or result.panel is None:
        return

    panel = result.panel
    D = panel.D
    eff = result.eff

    # ATT by calendar time
    time_ids = panel.time_ids
    cal_att = []
    for t in range(panel.T):
        treated_t = D[t, :] > 0
        if np.any(treated_t):
            cal_att.append(float(np.mean(eff[t, treated_t])))
        else:
            cal_att.append(np.nan)

    cal_att = np.array(cal_att)
    valid = ~np.isnan(cal_att)

    ax.plot(time_ids[valid], cal_att[valid], "o-", color="#2166ac",
            markersize=3, linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Calendar time")
    ax.set_ylabel("ATT")
