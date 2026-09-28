"""Plot the Stuart-number optimization runs produced by `optim.py`.

Usage:
    python runscripts/validation/eval_optim.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fluidgym import DEFAULT_PALETTE

OPTIM_DIR = Path("output/validation/optim")
CASES = ["hartmann", "shercliff", "hunt"]

PLOTS_DIR = Path("./paper/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FULL_WIDTH = 5.5
HALF_WIDTH = 0.49 * FULL_WIDTH
PANE_COLOR = "#F2F4F7"

mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",

        "pdf.fonttype": 42,  # TrueType in PDF
        "ps.fonttype": 42,  # TrueType in PS/EPS

        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        # Clean aesthetics
        "figure.autolayout": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

sns.set_style(
    style="darkgrid",
    rc={
        "axes.facecolor": PANE_COLOR,
    },
)
sns.set_palette(DEFAULT_PALETTE)

def load_optim_results() -> dict[str, pd.DataFrame]:
    results = {}
    for case in CASES:
        case_dir = OPTIM_DIR / case
        if not case_dir.exists():
            continue

        csv_path = case_dir / "optim_log.csv"
        if not csv_path.exists():
            continue

        optim_df = pd.read_csv(csv_path)
        results[case] = optim_df

    return results


if __name__ == "__main__":
    results = load_optim_results()

    fig, axs = plt.subplots(2, len(results), figsize=(FULL_WIDTH, 2.3), sharex="col")

    for i, (case, df) in enumerate(results.items()):
        target_N = df["target_N"].iloc[0]
        sns.lineplot(data=df, x="step", y="N_learn", ax=axs[0, i], zorder=2, label=r"Learned $\hat N$")
        axs[0, i].set_title(case.capitalize())
        axs[0, i].set_xlabel("Optimization Step")
        axs[0, i].set_ylabel(r"Stuart Number")
        axs[0, i].legend().remove()

        axs[0, i].axhline(target_N, color="gray", linestyle="--", zorder=1, label=r"Reference $N$")

        sns.lineplot(data=df, x="step", y="loss", ax=axs[1, i])
        axs[1, i].set_xlabel("Optimization Step")
        axs[1, i].set_ylabel("Loss")
        axs[1, i].set_yscale("log")
        axs[1, i].set_yticks([1e3,1e-1,1e-5])

    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        fancybox=False,
        shadow=False,
        frameon=False
    )
    fig.subplots_adjust(
        top=1.0,
        bottom=0.15,
        left=0.0,
        right=0.99,
    )
    plt.savefig(PLOTS_DIR / "optim_results.pdf", format="pdf", bbox_inches="tight")