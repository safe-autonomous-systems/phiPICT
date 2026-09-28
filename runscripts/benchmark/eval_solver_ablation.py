"""Aggregate the solver ablation timings into the paper figure and table.

Reads the outputs of ``ablate_solver_improvements.py`` from
``<results-root>/<env-id>/`` (default ``output/solver_ablation``). Results of the
earlier ablation (variants ``mhd_pict``, ``pict_full``, ...) can still be read.

Usage:
    python runscripts/benchmark/eval_solver_ablation.py \
        --env-id HartmannSmall3D-downward-easy-v0
    python runscripts/benchmark/eval_solver_ablation.py --env-id CylinderJet3D-hard-v0 \
        --results-root output_bkp/solver_ablation
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

from fluidgym import DEFAULT_PALETTE

FULL_WIDTH = 5.5
HALF_WIDTH = 0.49 * FULL_WIDTH
PANE_COLOR = "#F2F4F7"

BASELINE_VARIANT = "pict"

# The full phiPICT solver; the other two are its names in the earlier ablation
# (MHD, then non-MHD)
REFERENCE_VARIANTS = ["phipict", "mhd_pict", "pict_full"]

# Shift x-tick labels to the right in the small plot
TICK_LABEL_SHIFT = 0.14

VARIANT_ORDER = [
    "pict",
    "phipict",
    "pict_amg",
    "pict_intermediate_tol",
    "pict_warm_start",
    "no_amg",
    "no_intermediate_tol",
    "no_warm_start",
    "pict_full",
    "mhd_pict",
]

mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 500,
    }
)

sns.set_style(
    style="darkgrid",
    rc={
        "axes.facecolor": PANE_COLOR,
    },
)
sns.set_palette(DEFAULT_PALETTE)

PLOTS_DIR = Path("./paper/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path("./paper/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT = Path("./output/solver_ablation")


def load_timings(results_root: Path, env_id: str) -> pd.DataFrame:
    """Load the per-repeat timings of one ablation run."""
    csv_path = results_root / env_id / "solver_ablation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No ablation results at {csv_path}. Run "
            f"`python runscripts/benchmark/ablate_solver_improvements.py "
            f"env_id={env_id}` first."
        )
    return pd.read_csv(csv_path)


def load_convergence(results_root: Path, env_id: str) -> pd.DataFrame:
    """Load the solver diagnostics, if the ablation recorded them."""
    csv_path = results_root / env_id / "solver_convergence.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def attach_convergence(summary: pd.DataFrame, conv: pd.DataFrame) -> pd.DataFrame:
    """Fold the per-tag convergence rows into one flag per variant."""
    if conv.empty:
        summary["converged"] = True
        summary["n_not_converged"] = 0
        return summary

    # Only the forward is benchmarked, so old runs' adjoint rows are dropped
    conv = conv[conv["pass"] == "forward"]

    stats = conv.groupby("variant", sort=False).agg(
        converged=("converged", "all"),
        n_solves=("n_solves", "sum"),
        n_not_converged=("n_not_converged", "sum"),
    )

    epot = conv[conv["tag"] == "epot"]
    if not epot.empty:
        stats["epot_iters_mean"] = epot.groupby("variant", sort=False)[
            "iters_mean"
        ].mean()

    # ``iters_mean * n_solves`` is the exact iteration sum of that tag, so
    # summing over the tags gives every CG iteration the step ran
    conv = conv.assign(iters_total=conv["iters_mean"] * conv["n_solves"])
    stats["fwd_iters_total"] = conv.groupby("variant", sort=False)["iters_total"].sum()

    return summary.merge(stats.reset_index(), on="variant", how="left")


def order_variants(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort the variants by speedup, so the bars fall from left to right.

    The slowest variant is the baseline, so this also reads PICT left,
    $\\phi$-PICT right. Without the timings ``VARIANT_ORDER`` is the fallback.
    """
    if "forward_slowdown" in summary.columns:
        return summary.sort_values(
            "forward_slowdown", ascending=True, kind="stable"
        ).reset_index(drop=True)

    rank = {name: i for i, name in enumerate(VARIANT_ORDER)}
    return (
        summary.assign(
            _rank=[rank.get(v, len(VARIANT_ORDER)) for v in summary["variant"]]
        )
        .sort_values("_rank", kind="stable")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def reference_variant(summary: pd.DataFrame, requested: str | None) -> str | None:
    """The variant the speedups are measured against, if it is in the results."""
    if requested is not None:
        return requested

    present = set(summary["variant"])
    for name in REFERENCE_VARIANTS:
        if name in present:
            return name

    return None


def aggregate(df: pd.DataFrame, requested_reference: str | None = None) -> pd.DataFrame:
    """One row per variant, in the order the ablation ran them."""
    measured = df[(df["phase"] == "measure") & (df["pass"] == "forward")]

    rows = []
    for name, group in measured.groupby("variant", sort=False):
        fwd = group["forward_s"]

        row = {
            "variant": name,
            "label": group["label"].iloc[0],
            "piso_steps": float(group["piso_steps"].mean()),
            "forward_mean": float(fwd.mean()),
            "forward_std": float(fwd.std(ddof=1)) if len(fwd) > 1 else 0.0,
        }
        # recorded since the ablation covers the phiPICT solver switches
        if "peak_mem_gib" in group:
            row["peak_mem_gib"] = float(group["peak_mem_gib"].max())
        rows.append(row)

    summary = pd.DataFrame(rows)

    reference = reference_variant(summary, requested_reference)
    if reference is not None and reference in set(summary["variant"]):
        ref = summary[summary["variant"] == reference].iloc[0]

        summary["forward_speedup"] = ref["forward_mean"] / summary["forward_mean"]
        summary["forward_slowdown"] = summary["forward_mean"] / ref["forward_mean"]

    return summary


def variant_bars(
    ax: plt.Axes,
    values: np.ndarray,
    err: np.ndarray | None = None,
) -> None:
    """Draw one bar per variant, each in its own colour."""
    palette = sns.color_palette(n_colors=len(values))
    x = np.arange(len(values))

    ax.bar(
        x,
        values,
        0.7,
        yerr=err,
        color=palette,
        capsize=2,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        zorder=2,
    )
    ax.set_xticks(x)


def variant_labels(summary: pd.DataFrame) -> list[str]:
    """Tick labels, marking the variants that never reached the tolerance."""
    return [
        f"{label}$^\\dagger$" if n_nc > 0 else label
        for label, n_nc in zip(
            summary["label"], summary.get("n_not_converged", [0] * len(summary))
        )
    ]


def timing_panel(
    ax: plt.Axes, summary: pd.DataFrame, speedup_fontsize: float = 9
) -> None:
    """Forward wall-clock per step, annotated with the speedup over the baseline."""
    variant_bars(
        ax,
        summary["forward_mean"].to_numpy(),
        summary["forward_std"].to_numpy(),
    )
    ax.set_ylabel("Time per Step [s]")
    if "forward_slowdown" in summary:
        for xi, speedup in enumerate(summary["forward_slowdown"]):
            if np.isfinite(speedup):
                ax.annotate(
                    rf"$\times{speedup:.1f}$",
                    (xi, summary["forward_mean"].iloc[xi]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    va="bottom",
                    fontsize=speedup_fontsize,
                )
        ax.set_ylim(0.0, 1.28 * float(summary["forward_mean"].max()))


def iteration_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Total forward CG iterations, the hardware-independent version of the timings."""
    n = len(summary)
    variant_bars(
        ax,
        summary.get("fwd_iters_total", pd.Series(np.full(n, np.nan))).to_numpy(),
    )
    ax.set_ylabel("Total CG iterations per step")


def shift_tick_labels(ax: plt.Axes, dx: float = TICK_LABEL_SHIFT) -> None:
    """Slide the x tick labels right, so a rotated one sits under its bar."""
    offset = ScaledTranslation(dx, 0.0, ax.figure.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)


def variant_legend(fig: plt.Figure, summary: pd.DataFrame, ncol: int, y: float) -> None:
    """Name the variants below the panels, in the colours the bars use."""
    palette = sns.color_palette(n_colors=len(summary))
    fig.legend(
        [Patch(facecolor=c) for c in palette],
        variant_labels(summary),
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fancybox=False,
        shadow=False,
        frameon=False,
    )


def plot_ablation(summary: pd.DataFrame, out_path: Path) -> None:
    """Wall-clock and iteration count, one panel each."""
    fig, axs = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.0))
    # The panels are half as wide as in the standalone plot, so the annotations
    # of the two fastest variants would otherwise run into each other
    timing_panel(axs[0], summary, speedup_fontsize=6)
    iteration_panel(axs[1], summary)

    # The variants are named once, in the legend, rather than under both panels
    for ax in axs:
        ax.tick_params(axis="x", labelbottom=False, length=0)

    variant_legend(fig, summary, ncol=3, y=0.06)
    fig.subplots_adjust(top=0.97, bottom=0.06, left=0.09, right=0.99, wspace=0.45)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_timing(summary: pd.DataFrame, out_path: Path) -> None:
    """The wall-clock panel on its own, square, for use as a half-width figure."""
    fig, ax = plt.subplots(1, 1, figsize=(HALF_WIDTH, 2.0))
    timing_panel(ax, summary)
    ax.set_xticklabels(variant_labels(summary), rotation=30, ha="right")
    shift_tick_labels(ax)

    fig.subplots_adjust(top=0.97, bottom=0.3, left=0.18, right=0.98)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def print_convergence_table(summary: pd.DataFrame) -> None:
    """Print the non-converged solve counts, which no longer go into the figure."""
    if "n_not_converged" not in summary.columns:
        return

    n = len(summary)
    table = pd.DataFrame(
        {
            "label": summary["label"],
            "n_solves": summary.get("n_solves", pd.Series(np.full(n, np.nan))),
            "n_not_converged": summary["n_not_converged"],
        }
    )
    print("\nSolves that exhausted their iteration cap:")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(table.to_string(index=False))


def write_table(summary: pd.DataFrame, out_path: Path) -> None:
    """Write the ablation as a booktabs table."""
    has_iters = "epot_iters_mean" in summary.columns
    header = r"Solver & Time per Step [s] & Speedup \\"
    if has_iters:
        header = r"Solver & CG its.\ ($\phi$) & Time per Step [s] & Speedup \\"

    lines = [
        r"\begin{tabular}{l" + ("c" * (3 if has_iters else 2)) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    stalled = False
    for _, row in summary.iterrows():
        speedup = row.get("forward_speedup", float("nan"))
        label = row["label"]
        if row.get("n_not_converged", 0) > 0:
            label += r"$^\dagger$"
            stalled = True

        cells = [label]
        if has_iters:
            cells.append(f"${row['epot_iters_mean']:.0f}$")
        cells += [
            f"${row['forward_mean']:.3f} \\pm {row['forward_std']:.3f}$",
            f"$\\times{speedup:.2f}$",
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    if stalled:
        # The reader has to be told what the marked rows mean, and it is a claim
        # in our favour: those variants never reach the tolerance at all
        lines += [
            r"% $^\dagger$ did not converge: the solve exhausts its iteration "
            r"cap at a residual above the tolerance, so the reported time is a "
            r"lower bound on the cost of an equal-accuracy solve.",
        ]
    lines += [""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="HartmannSmall3D-downward-easy-v0")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Directory holding one ablation output directory per env id.",
    )
    parser.add_argument("--plot-name", default="solver_ablation.pdf")
    parser.add_argument("--timing-plot-name", default="solver_ablation_timing.pdf")
    parser.add_argument("--table-name", default="solver_ablation_table.tex")
    parser.add_argument(
        "--reference-variant",
        default=None,
        help=(
            "Variant the speedups are measured against. Defaults to the full "
            "solver of whichever ablation the results come from."
        ),
    )
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary = order_variants(
        attach_convergence(
            aggregate(
                load_timings(args.results_root, args.env_id), args.reference_variant
            ),
            load_convergence(args.results_root, args.env_id),
        )
    )
    if summary.empty:
        raise RuntimeError("The ablation results contain no measured repeats.")

    plot_path = PLOTS_DIR / args.plot_name
    timing_plot_path = PLOTS_DIR / args.timing_plot_name
    table_path = TABLES_DIR / args.table_name

    plot_ablation(summary, plot_path)
    plot_timing(summary, timing_plot_path)
    write_table(summary, table_path)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(
            summary[
                [
                    "label",
                    "piso_steps",
                    "forward_mean",
                    "forward_std",
                ]
                + (["forward_speedup"] if "forward_speedup" in summary else [])
                + (["peak_mem_gib"] if "peak_mem_gib" in summary else [])
            ].to_string(index=False)
        )

    print_convergence_table(summary)

    print(f"\nWrote {plot_path}, {timing_plot_path} and {table_path}.")


if __name__ == "__main__":
    main()
