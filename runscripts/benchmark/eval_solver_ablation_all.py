"""Speedup of phiPICT over PICT for every environment with ablation results.

Reads every ``<results-root>/<env-id>/`` written by ``ablate_solver_improvements.py``
and plots one bar per environment: the time per env step of PICT divided by that of
phiPICT, with the standard deviation propagated from both timings.

Usage:
    python runscripts/benchmark/eval_solver_ablation_all.py
    python runscripts/benchmark/eval_solver_ablation_all.py --exclude Hartmann
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from eval_solver_ablation import (
    BASELINE_VARIANT,
    FULL_WIDTH,
    PLOTS_DIR,
    REFERENCE_VARIANTS,
    RESULTS_ROOT,
    TABLES_DIR,
    aggregate,
    attach_convergence,
    load_convergence,
    load_timings,
)
from matplotlib.patches import Patch


def env_family(env_id: str) -> str:
    """The environment family, e.g. ``CylinderJet3D-hard-v0`` -> ``CylinderJet``."""
    match = re.match(r"([A-Za-z]+?)(Small|Medium|Large)?\d?D", env_id)
    return match.group(1) if match else env_id


DIFFICULTIES = {"easy": 0, "medium": 1, "hard": 2}


def env_sort_key(env_id: str) -> tuple[str, int]:
    """Sort by the env id without its difficulty, then easy, medium, hard."""
    for name, rank in DIFFICULTIES.items():
        if f"-{name}-" in env_id:
            return env_id.replace(f"-{name}-", "-"), rank
    return env_id, len(DIFFICULTIES)


def env_label(env_id: str) -> str:
    """Short tick label: the env id without the version suffix."""
    return re.sub(r"-v\d+$", "", env_id)


def speedup_row(results_root: Path, env_id: str) -> dict | None:
    """PICT and phiPICT timings of one env and the speedup, or None if incomplete."""
    summary = attach_convergence(
        aggregate(load_timings(results_root, env_id), BASELINE_VARIANT),
        load_convergence(results_root, env_id),
    )
    present = set(summary["variant"])
    reference = next((v for v in REFERENCE_VARIANTS if v in present), None)
    if BASELINE_VARIANT not in present or reference is None:
        return None

    base = summary[summary["variant"] == BASELINE_VARIANT].iloc[0]
    ours = summary[summary["variant"] == reference].iloc[0]
    speedup = base["forward_mean"] / ours["forward_mean"]
    # relative errors add in quadrature for a quotient
    speedup_std = speedup * np.hypot(
        base["forward_std"] / base["forward_mean"],
        ours["forward_std"] / ours["forward_mean"],
    )
    return {
        "env_id": env_id,
        "family": env_family(env_id),
        "pict_s": base["forward_mean"],
        "pict_std_s": base["forward_std"],
        "phipict_s": ours["forward_mean"],
        "phipict_std_s": ours["forward_std"],
        "speedup": speedup,
        "speedup_std": speedup_std,
        "pict_piso_steps": base["piso_steps"],
        "phipict_piso_steps": ours["piso_steps"],
        "pict_not_converged": int(base.get("n_not_converged", 0) or 0),
        "phipict_not_converged": int(ours.get("n_not_converged", 0) or 0),
    }


def collect(results_root: Path, exclude: list[str]) -> pd.DataFrame:
    """One row per environment with complete PICT and phiPICT results."""
    rows = []
    for env_dir in sorted(results_root.iterdir()):
        env_id = env_dir.name
        if not (env_dir / "solver_ablation.csv").is_file():
            continue
        if any(pattern in env_id for pattern in exclude):
            continue
        row = speedup_row(results_root, env_id)
        if row is None:
            print(f"Skipping {env_id}: PICT or phiPICT results missing.")
            continue
        rows.append(row)
    return pd.DataFrame(rows)


def plot_speedups(df: pd.DataFrame, out_path: Path) -> None:
    """One bar per env, coloured by env family, on a log axis."""
    families = list(dict.fromkeys(df["family"]))
    palette = dict(
        zip(families, sns.color_palette(n_colors=len(families)), strict=True)
    )
    x = np.arange(len(df))

    width = max(FULL_WIDTH, 0.22 * len(df))
    fig, ax = plt.subplots(figsize=(width, 2.4))
    ax.bar(
        x,
        df["speedup"],
        0.7,
        yerr=df["speedup_std"],
        color=[palette[f] for f in df["family"]],
        capsize=1.5,
        error_kw={"elinewidth": 0.6, "capthick": 0.6},
        zorder=2,
    )
    ax.axhline(1.0, color="black", linewidth=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel(r"Speedup over PICT")
    for xi, (speedup, std) in enumerate(
        zip(df["speedup"], df["speedup_std"], strict=True)
    ):
        # above the error bar
        ax.annotate(
            rf"${speedup:.1f}$",
            (xi, speedup + std),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=5,
        )
    # PICT solves that ran out of iterations: its time is a lower bound
    labels = [
        f"{env_label(env)}$^\\dagger$" if n > 0 else env_label(env)
        for env, n in zip(df["env_id"], df["pict_not_converged"], strict=True)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_ylim(top=2.0 * float((df["speedup"] + df["speedup_std"]).max()))

    fig.legend(
        [Patch(facecolor=palette[f]) for f in families],
        families,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(families),
        frameon=False,
    )
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_table(df: pd.DataFrame, out_path: Path) -> None:
    """Write the per-env timings and speedups as a booktabs table."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Environment & PICT [s] & $\phi$-PICT [s] & Speedup \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        env = env_label(row["env_id"])
        if row["pict_not_converged"] > 0:
            env += r"$^\dagger$"
        lines.append(
            f"{env} & ${row['pict_s']:.3f} \\pm {row['pict_std_s']:.3f}$ & "
            f"${row['phipict_s']:.3f} \\pm {row['phipict_std_s']:.3f}$ & "
            f"$\\times{row['speedup']:.1f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    if (df["pict_not_converged"] > 0).any():
        lines.append(
            r"% $^\dagger$ PICT solves exhausted their iteration cap, so its time "
            r"is a lower bound."
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Skip envs whose id contains any of these substrings.",
    )
    parser.add_argument("--plot-name", default="solver_ablation_speedups.pdf")
    parser.add_argument("--table-name", default="solver_ablation_speedups_table.tex")
    parser.add_argument("--csv-name", default="solver_ablation_speedups.csv")
    args = parser.parse_args()

    df = collect(args.results_root, args.exclude)
    if df.empty:
        raise RuntimeError(f"No complete ablation results under {args.results_root}.")
    # grouped by family, then by env, easy to hard
    df = df.sort_values(
        ["family", "env_id"],
        key=lambda col: col.map(env_sort_key) if col.name == "env_id" else col,
        kind="stable",
    ).reset_index(drop=True)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / args.plot_name
    table_path = TABLES_DIR / args.table_name
    csv_path = args.results_root / args.csv_name

    plot_speedups(df, plot_path)
    write_table(df, table_path)
    df.to_csv(csv_path, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(
            df[
                ["env_id", "pict_s", "phipict_s", "speedup", "speedup_std"]
                + ["pict_not_converged"]
            ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
    print(f"\nWrote {plot_path}, {table_path} and {csv_path}.")


if __name__ == "__main__":
    main()
