"""Bench eval artifact plots."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


def top_frac_rollout_mask(vals: np.ndarray, top_frac: float) -> np.ndarray:
    y = np.asarray(vals, dtype=np.float64).reshape(-1)
    frac = float(top_frac)
    if y.size == 0 or frac <= 0.0 or frac >= 1.0:
        return np.ones(y.size, dtype=np.bool_)
    k = max(1, int(np.ceil(y.size * frac)))
    order = np.argsort(-y)[:k]
    mask = np.zeros(y.size, dtype=np.bool_)
    mask[order] = True
    return mask


def save_best_occ_rollout_init_trace_png(
    out_path: str,
    vals: list,
    started_from_best: list[bool],
    title: str = "",
    *,
    top_frac: float | None = None,
) -> None:
    y = np.asarray(vals, dtype=np.float64).reshape(-1)
    flags = np.asarray(started_from_best, dtype=np.bool_)
    if y.size == 0:
        return
    n_total = int(y.size)
    if top_frac is not None:
        keep = top_frac_rollout_mask(y, top_frac)
        if not np.any(keep):
            return
        y = y[keep]
        flags = flags[keep]
        x = np.flatnonzero(keep).astype(np.int32) + 1
    else:
        x = np.arange(1, y.size + 1, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10, 4))
    m_best = flags
    m_other = ~flags
    if np.any(m_other):
        ax.scatter(
            x[m_other],
            y[m_other],
            s=22,
            c="#2980b9",
            marker="o",
            alpha=0.75,
            edgecolors="none",
            zorder=2,
        )
    if np.any(m_best):
        ax.scatter(
            x[m_best],
            y[m_best],
            s=55,
            c="#c0392b",
            marker="x",
            alpha=0.9,
            linewidths=1.35,
            zorder=3,
        )
    ax.set_xlabel("Rollout index")
    ax.set_ylabel("Best occupancy ratio")
    if title:
        ax.set_title(title)
    elif top_frac is not None and 0.0 < float(top_frac) < 1.0:
        pct = float(top_frac) * 100.0
        ax.set_title(
            f"Top {pct:g}% rollouts by best_occ "
            f"(n={y.size}/{n_total}, × = warm/elite init, ○ = otherwise)"
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def peak_log_likelihood_vector(
    num_episodes: int,
    *,
    peak_log_likelihood: list | None = None,
    peak_reward_log_likelihoods: list | None = None,
) -> np.ndarray | None:
    if peak_log_likelihood is not None:
        if len(peak_log_likelihood) != num_episodes:
            return None
        return np.array(
            [np.nan if v is None else float(v) for v in peak_log_likelihood],
            dtype=np.float64,
        )
    peak_ll = list(peak_reward_log_likelihoods or [])
    if len(peak_ll) != num_episodes:
        return None
    return np.array(
        [
            np.nan if entry.get("log_likelihood") is None else float(entry["log_likelihood"])
            for entry in peak_ll
        ],
        dtype=np.float64,
    )


def save_best_occ_vs_log_likelihood_png(
    out_path: str,
    best_occ_values: list,
    peak_reward_log_likelihoods: list | None = None,
    *,
    peak_log_likelihood: list | None = None,
    instance_index: int = -1,
    top_frac: float = 1.0,
    max_plot_points: int = 5000,
) -> bool:
    vals = np.asarray(best_occ_values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return False

    log_likelihoods = peak_log_likelihood_vector(
        vals.size,
        peak_log_likelihood=peak_log_likelihood,
        peak_reward_log_likelihoods=peak_reward_log_likelihoods,
    )
    if log_likelihoods is None:
        return False
    valid = np.isfinite(log_likelihoods)
    if not np.any(valid):
        return False

    elite_mask = top_frac_rollout_mask(vals, float(top_frac))
    plot_mask = elite_mask & valid
    if not np.any(plot_mask):
        return False

    x_all = log_likelihoods[plot_mask]
    y_all = vals[plot_mask]
    n_plot = int(plot_mask.sum())
    n_valid = int(valid.sum())
    frac = float(top_frac)
    scatter_color = sns.color_palette("flare")[2]

    cap = int(max_plot_points)
    if cap > 0 and n_plot > cap:
        pick = np.random.default_rng(0).choice(n_plot, size=cap, replace=False)
        x = x_all[pick]
        y = y_all[pick]
        n_shown = cap
    else:
        x = x_all
        y = y_all
        n_shown = n_plot

    sns.set_theme(style="ticks", context="notebook", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=x,
        y=y,
        ax=ax,
        s=36,
        alpha=0.75,
        color=scatter_color,
        label="Rollouts",
    )
    if n_plot >= 2:
        use_regplot_ci = n_shown <= 5000
        if use_regplot_ci:
            sns.regplot(
                x=x,
                y=y,
                ax=ax,
                scatter=False,
                color="0.35",
                ci=95,
                line_kws={"linewidth": 1.2, "alpha": 0.85, "label": "Linear fit"},
            )
            ci_patch = mpatches.Patch(
                facecolor="0.75",
                edgecolor="none",
                alpha=0.35,
                label="95% confidence interval",
            )
        else:
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(
                x_line,
                np.polyval(coef, x_line),
                color="0.35",
                linewidth=1.2,
                alpha=0.85,
                label="Linear fit",
            )
            ci_patch = None
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=scatter_color,
                markersize=8,
                alpha=0.75,
                label="Rollouts",
            ),
            Line2D([0], [0], color="0.35", linewidth=1.2, label="Linear fit"),
        ]
        if ci_patch is not None:
            legend_handles.append(ci_patch)
        ax.legend(handles=legend_handles, loc="best", frameon=True)
    else:
        ax.legend(loc="best", frameon=True)

    ax.set_xlabel("Peak log-likelihood (γ-discounted, last 6 pre-peak actions)")
    ax.set_ylabel("Best occupancy ratio")
    if frac >= 1.0:
        subset = f"all rollouts (n={n_plot}/{vals.size}"
    else:
        subset = f"top {frac * 100.0:g}% (n={n_plot}/{vals.size}"
    sample_note = f", plot sample {n_shown}" if n_shown < n_plot else ""
    ax.set_title(
        f"Instance {instance_index} — best_occ vs log-likelihood "
        f"({subset}{sample_note}, {n_valid} with finite LL)"
    )
    ax.grid(True, alpha=0.25)
    sns.despine(ax=ax, offset=4, trim=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
