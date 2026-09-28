"""Time PICT against phiPICT (and any other combination of the solver switches).

Every variant (``runscripts/configs/<variant_dir>/<name>.yaml``) sets every switch of
the solver stack, see ``solver_ablation/pict.yaml`` (the original PICT) and
``solver_ablation/phipict.yaml`` (all features). The environment and its solver
tolerances are the same for every variant.

Usage:
    python runscripts/benchmark/ablate_solver_improvements.py
    python runscripts/benchmark/ablate_solver_improvements.py ablation=cylinder
    python runscripts/benchmark/ablate_solver_improvements.py \
        env_id=HartmannMedium3D-downward-easy-v0
"""

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf

import fluidgym
from fluidgym import DEFAULT_PALETTE
from fluidgym.envs.mhd.mhd_env import MHDEnv
from fluidgym.registry import registry
from phipict import _C
from phipict.core.piso_simulation import Simulation as PISOSimulation
from phipict.solvers import amg
from phipict.solvers.amg import is_pyamg_available
from phipict.solvers.stats import SolverStatsRecorder
from phipict.solvers.tolerance import SolverTolerance

local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
fluidgym.config.update("local_data_path", local_data_path)

logger = logging.getLogger("solver_ablation")

# Resolved from the file, not from the working directory: hydra chdirs into the
# run directory before main() is entered, so a relative path would not survive
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

mpl.rcParams.update(
    {
        "text.usetex": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

sns.set_style(
    style="darkgrid",
    rc={
        "axes.facecolor": "#F2F4F7",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.bottom": True,
        "ytick.left": True,
    },
)
sns.set_palette(DEFAULT_PALETTE)

# The switches every variant file sets. `intermediate_tol` switches the env's
# `pressure_tol_intermediate` (from the ablation config) on or off.
SWITCHES = [
    "fused_krylov",
    "krylov_graphs",
    "cg_preconditioner",
    "bicg_preconditioner",
    "native_amg",
    "pressure_amg",
    "share_amg_interpolation",
    "amg_fp32_vcycle",
    "coarse_refresh_interval",
    "intermediate_tol",
    "pressure_warm_start",
    "potential_amg",
    "potential_reuse_result",
]
# Only meaningful for MHD envs, which solve for the electric potential
MHD_SWITCHES = ["potential_amg", "potential_reuse_result"]

FORCED_ENV_KWARGS: dict[str, Any] = {
    "load_initial_domain": True,
    "load_domain_statistics": False,
    "randomize_initial_state": False,
}


def free_memory() -> None:
    """Drop cached allocations so the next variant starts from a clean slate."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def elapsed_since(t0: float) -> float:
    """Seconds since ``t0``, with the CUDA queue drained first.

    ``perf_counter`` measures the host, so without the synchronisation it would
    time the kernel launches rather than the solves.
    """
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def is_mhd_env(env_id: str) -> bool:
    """Whether ``env_id`` is an MHD environment, i.e. whether AMG applies.

    Read off the registry rather than from a constructed env, so the answer is
    available before any solver setting has to be chosen.
    """
    if env_id not in registry.env_specs:
        raise ValueError(f"Environment {env_id} is not registered.")
    entry_point = registry.env_specs[env_id].entry_point
    return isinstance(entry_point, type) and issubclass(entry_point, MHDEnv)


def load_variants(cfg: DictConfig) -> list[DictConfig]:
    """Load the requested variant configs from the variant directory."""
    variant_dir = CONFIG_DIR / str(cfg.variant_dir)
    variants = []

    for name in cfg.variants:
        path = variant_dir / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"Unknown ablation variant {name!r}: {path} does not exist. "
                f"Available: {sorted(p.stem for p in variant_dir.glob('*.yaml'))}"
            )
        variant = OmegaConf.load(path)
        assert isinstance(variant, DictConfig)
        if variant.get("name", None) != name:
            raise ValueError(
                f"Variant file {path} declares name {variant.get('name', None)!r}, "
                f"which does not match its filename."
            )
        missing = [k for k in SWITCHES if k not in variant]
        if missing:
            raise ValueError(f"Variant file {path} does not set {missing}.")
        variants.append(variant)

    return variants


def resolve_env_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """The env kwargs, the same for every variant.

    ``shared_solver`` (iteration caps) comes after the base kwargs and the forced
    ones last. The solver switches are not env kwargs: they are global settings of
    phipict or attributes of the simulation, set in :func:`make_env`, so the
    ablation works for any environment.
    """
    base = OmegaConf.to_container(cfg.env_kwargs, resolve=True)
    assert isinstance(base, dict)
    shared = OmegaConf.to_container(cfg.shared_solver, resolve=True)
    assert isinstance(shared, dict)

    return {
        **{str(k): v for k, v in base.items()},
        **{str(k): v for k, v in shared.items()},
        **FORCED_ENV_KWARGS,
    }


def amg_options(variant: DictConfig) -> dict[str, Any]:
    """Keyword arguments of `amg.hierarchy_for` for this variant."""
    return {
        "reuse_interpolation": bool(variant.share_amg_interpolation),
        "native_dtype": torch.float32 if bool(variant.amg_fp32_vcycle) else None,
    }


def apply_global_settings(variant: DictConfig) -> None:
    """Process-wide switches: linear solver backend and AMG implementation."""
    _C.SetKrylovSettings(
        enabled=bool(variant.fused_krylov),
        useGraphs=bool(variant.krylov_graphs),
        cgPreconditioner=str(variant.cg_preconditioner),
        bicgPreconditioner=str(variant.bicg_preconditioner),
    )
    amg.USE_NATIVE = bool(variant.native_amg)
    # every variant starts without cached AMG setups (in memory); the on-disk cache
    # (amg_cache_dir) only saves the host-side setup and does not change the solves
    amg.clear_interpolation_cache()


# `pressure_tol_intermediate: auto` loosens the env's final pressure tolerance by
# this factor, the ratio the MHD envs use (rtol 1e-2 -> 1e-1)
AUTO_INTERMEDIATE_FACTOR = 10.0


def intermediate_tolerance(cfg: DictConfig, sim: Any) -> SolverTolerance | float:
    """The pressure tolerance of all but the last corrector.

    Either given in the config (``{rtol, atol}``) or ``auto``: the env's final
    pressure tolerance times ``AUTO_INTERMEDIATE_FACTOR``, so one config serves
    every environment.
    """
    tol = cfg.pressure_tol_intermediate
    if isinstance(tol, str):
        if tol != "auto":
            raise ValueError(
                f"pressure_tol_intermediate must be a mapping or 'auto', got {tol!r}."
            )
        final = sim.pressure_tol
        f = AUTO_INTERMEDIATE_FACTOR
        if isinstance(final, SolverTolerance):
            return SolverTolerance(
                rtol=None if final.rtol is None else f * final.rtol,
                atol=None if final.atol is None else f * final.atol,
            )
        return f * float(final)
    tol_dict = OmegaConf.to_container(tol, resolve=True)
    assert isinstance(tol_dict, dict)
    return SolverTolerance(**tol_dict)


def apply_sim_settings(cfg: DictConfig, sim: Any, variant: DictConfig) -> None:
    """Per-simulation switches, set on the simulation the environment built."""
    if bool(variant.intermediate_tol):
        sim.pressure_tol_intermediate = intermediate_tolerance(cfg, sim)
        logger.info(
            "  intermediate pressure tolerance: %s (final: %s)",
            sim.pressure_tol_intermediate,
            sim.pressure_tol,
        )
    else:
        sim.pressure_tol_intermediate = None
    sim.pressure_warm_start = bool(variant.pressure_warm_start)
    sim.pressure_use_amg = (
        None if variant.pressure_amg is None else bool(variant.pressure_amg)
    )
    sim._pressure_amg_options = {
        **amg_options(variant),
        "coarse_refresh_interval": int(variant.coarse_refresh_interval),
    }
    sim._pressure_amg_hierarchy = None
    sim._pressure_amg_fingerprint = None
    if hasattr(sim, "_potential_use_preconditioner"):  # MHD
        sim._potential_use_preconditioner = bool(variant.potential_amg)
        sim._potential_reuse_result = bool(variant.potential_reuse_result)
        sim._potential_amg_options = amg_options(variant)
        sim._epot_amg_hierarchy = None
        sim._epot_amg_matrix = None


def make_env(
    cfg: DictConfig, variant: DictConfig, env_kwargs: dict[str, Any], step_length: float
):
    """Create the benchmark environment for one variant, with its switches applied."""
    apply_global_settings(variant)
    # reset() builds the simulation and runs its first steps before the variant's
    # switches can be set on it; without this, the automatic pressure AMG would run
    # a host-side AMG setup for variants that do not use it
    auto_min_rows = PISOSimulation.PRESSURE_AMG_AUTO_MIN_ROWS
    if variant.pressure_amg is not None and not bool(variant.pressure_amg):
        PISOSimulation.PRESSURE_AMG_AUTO_MIN_ROWS = 2**62
    try:
        env = fluidgym.make(
            cfg.env_id,
            **env_kwargs,
            step_length=step_length,
            differentiable=False,
        )
        env.seed(cfg.seed)
        env.reset()
    finally:
        PISOSimulation.PRESSURE_AMG_AUTO_MIN_ROWS = auto_min_rows
    logger.debug(
        "  initial domain: %s (mode %s)",
        env.initial_domain_id,
        env.mode.value,
    )
    apply_sim_settings(cfg, env._sim, variant)

    # An env step covers ``step_length`` of physical time. With adaptive
    # substepping the PISO step count behind it is set by the CFL condition of
    # the current flow, so it could differ between variants and the wall-clock
    # comparison would no longer be like for like. Pinning one PISO step per
    # sim step makes the work exactly ``n_piso_steps`` per env step everywhere
    if cfg.fixed_substeps:
        env._sim.substeps = 1

    return env


def make_actions(cfg: DictConfig, env: Any, n: int) -> list[torch.Tensor]:
    """The action sequence, the same for every variant.

    ``random``: uniformly random actions from a dedicated generator (not the env's
    RNG, so the sequence does not depend on how much randomness the env consumed).
    ``zero``: the uncontrolled rollout.
    """
    if str(cfg.actions) == "zero":
        return [env._zero_action] * n
    space = env.action_space
    low = torch.as_tensor(space.low, dtype=torch.float32)
    high = torch.as_tensor(space.high, dtype=torch.float32)
    gen = torch.Generator().manual_seed(int(cfg.seed))
    shape = env._zero_action.shape
    return [
        (low + (high - low) * torch.rand(shape, generator=gen)).to(
            device=env._zero_action.device, dtype=env._zero_action.dtype
        )
        for _ in range(n)
    ]


def measure_convergence(
    cfg: DictConfig,
    variant: DictConfig,
    env_kwargs: dict[str, Any],
    step_length: float,
) -> list[dict[str, Any]]:
    """Record whether the linear solves converged, in a separate untimed pass.

    A wall-clock comparison only means something if every variant solves to the
    same accuracy, and at the env's production tolerances the unpreconditioned
    potential solve does not: it stagnates above the target residual and returns
    its best iterate after exhausting ``potential_solve_max_iter``. One row per
    solve tag, so the timing table can say which variants got there and which
    ran out of iterations.

    The recorder synchronises the device around every solve, so it must not run
    inside the timed loop; this pass is separate and its wall time is ignored.
    """
    free_memory()
    env = make_env(cfg, variant, env_kwargs, step_length)
    actions = make_actions(cfg, env, int(cfg.convergence_check_steps))

    with SolverStatsRecorder(time_solves=False) as rec:
        for action in actions:
            with torch.no_grad():
                env.step(action)

    stats = rec.summary()
    rows: list[dict[str, Any]] = []

    for tag in rec.tags:
        n_solves = stats[f"solver/{tag}/n_solves"]
        n_not_converged = stats[f"solver/{tag}/n_not_converged"]
        rows.append(
            {
                **variant_fields(variant),
                "pass": "forward",
                "tag": tag,
                "converged": n_not_converged == 0.0,
                "n_solves": int(n_solves),
                "n_not_converged": int(n_not_converged),
                "iters_mean": stats[f"solver/{tag}/iters_mean"],
                "iters_max": stats[f"solver/{tag}/iters_max"],
                "rel_res_max": stats[f"solver/{tag}/rel_res_max"],
                "tol": stats[f"solver/{tag}/tol_max"],
            }
        )

        if n_not_converged > 0.0:
            logger.warning(
                "  [%s] %s: %d of %d solves did not converge, worst "
                "relative residual %.03e at tolerance %.03e.",
                variant.label,
                tag,
                int(n_not_converged),
                int(n_solves),
                stats[f"solver/{tag}/rel_res_max"],
                stats[f"solver/{tag}/tol_max"],
            )

    del env
    free_memory()

    return rows


def measure_forward(
    cfg: DictConfig, variant: DictConfig, env_kwargs: dict[str, Any], step_length: float
) -> list[dict[str, Any]]:
    """Time forward-only env steps, i.e. the cost of a rollout without gradients."""
    free_memory()
    env = make_env(cfg, variant, env_kwargs, step_length)
    actions = make_actions(cfg, env, int(cfg.warmup_steps) + int(cfg.repeats))
    sim = env._sim

    rows: list[dict[str, Any]] = []

    # The first step of an AMG variant pays for the host-side hierarchy setup,
    # which is a one-off amortised over the whole run. It is reported as its own
    # row rather than folded into the steady-state numbers
    for i in range(int(cfg.warmup_steps)):
        piso_before = sim.total_step
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            env.step(actions[i])
        rows.append(
            {
                **variant_fields(variant),
                "pass": "forward",
                "phase": "warmup",
                "repeat": i,
                "piso_steps": sim.total_step - piso_before,
                "forward_s": elapsed_since(t0),
            }
        )

    for r in range(int(cfg.repeats)):
        piso_before = sim.total_step
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            env.step(actions[int(cfg.warmup_steps) + r])
        forward_s = elapsed_since(t0)

        rows.append(
            {
                **variant_fields(variant),
                "pass": "forward",
                "phase": "measure",
                "repeat": r,
                "piso_steps": sim.total_step - piso_before,
                "forward_s": forward_s,
                "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        logger.info("  [FWD] repeat %d: %.4f s", r, forward_s)

    del env
    free_memory()

    return rows


def variant_fields(variant: DictConfig) -> dict[str, Any]:
    """The identifying columns every measurement row carries."""
    return {
        "variant": str(variant.name),
        "label": str(variant.label),
        # null (the automatic pressure AMG) is written as "auto"
        **{k: "auto" if variant[k] is None else variant[k] for k in SWITCHES},
    }


def oom_row(variant: DictConfig) -> dict[str, Any]:
    """Placeholder row for a variant that ran out of memory."""
    return {
        **variant_fields(variant),
        "pass": "forward",
        "phase": "oom",
        "repeat": -1,
        "piso_steps": float("nan"),
        "forward_s": float("nan"),
        "peak_mem_gib": float("nan"),
    }


def summarize(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Aggregate the per-repeat timings into one row per variant."""
    measured = df[df["phase"] == "measure"]
    rows: list[dict[str, Any]] = []

    for name, group in measured.groupby("variant", sort=False):
        fwd = group["forward_s"]

        rows.append(
            {
                "variant": name,
                "label": group["label"].iloc[0],
                **{k: group[k].iloc[0] for k in SWITCHES},
                "piso_steps_per_env_step": float(group["piso_steps"].mean()),
                "forward_mean_s": float(fwd.mean()),
                "forward_std_s": float(fwd.std(ddof=1)) if len(fwd) > 1 else 0.0,
                "peak_mem_gib": float(group["peak_mem_gib"].max()),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    baseline = str(cfg.baseline_variant)
    if baseline in set(summary["variant"]):
        ref = summary[summary["variant"] == baseline].iloc[0]
        # Speedup over the plain PICT baseline: what the improvements buy
        summary["forward_speedup"] = ref["forward_mean_s"] / summary["forward_mean_s"]

    full = str(cfg.reference_variant)
    if full in set(summary["variant"]):
        ours = summary[summary["variant"] == full].iloc[0]
        # Slowdown against the full configuration: the marginal cost of dropping
        # the one component this variant leaves out
        summary["forward_slowdown_vs_full"] = (
            summary["forward_mean_s"] / ours["forward_mean_s"]
        )

    return summary


def plot_summary(summary: pd.DataFrame, cfg: DictConfig, plot_file: str) -> None:
    """One bar per variant, coloured by variant, of the forward cost."""
    if summary.empty:
        logger.warning("No successful measurements to plot.")
        return

    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        x,
        summary["forward_mean_s"],
        0.7,
        yerr=summary["forward_std_s"],
        color=sns.color_palette(n_colors=len(summary)),
        capsize=3,
        zorder=2,
    )

    if "forward_speedup" in summary:
        for xi, speedup in enumerate(summary["forward_speedup"]):
            if np.isfinite(speedup):
                ax.text(
                    xi,
                    0.0,
                    rf"$\times{speedup:.2f}$",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # A variant whose solves ran out of iterations is timed at a different
    # accuracy from the rest, so the bar is marked rather than left to be read
    # as a like-for-like number
    labels = list(summary["label"])
    if "n_not_converged" in summary:
        labels = [
            f"{label}$^\\dagger$" if n > 0 else label
            for label, n in zip(labels, summary["n_not_converged"])
        ]

    ax.set_xlabel("")
    ax.set_ylabel(rf"Time per env step ({int(cfg.n_piso_steps)} PISO steps) [s]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    fig.subplots_adjust(top=0.95, bottom=0.24, left=0.12, right=0.98)
    plt.savefig(plot_file, format="pdf")
    plt.close(fig)


def attach_convergence(summary: pd.DataFrame, conv: pd.DataFrame) -> pd.DataFrame:
    """Fold the per-tag convergence rows into one flag per variant."""
    if summary.empty or conv.empty:
        return summary

    stats = conv.groupby("variant", sort=False).agg(
        converged=("converged", "all"),
        n_solves=("n_solves", "sum"),
        n_not_converged=("n_not_converged", "sum"),
    )

    return summary.merge(stats.reset_index(), on="variant", how="left")


def log_summary(summary: pd.DataFrame) -> None:
    """Print the table in the shape the paper reports it in."""
    if summary.empty:
        return

    logger.info("=" * 72)
    logger.info(
        "%-24s %14s %9s %12s %10s",
        "variant",
        "FWD [s]",
        "speedup",
        "not conv.",
        "peak mem.",
    )
    logger.info("-" * 72)
    for _, row in summary.iterrows():
        speedup = row.get("forward_speedup", float("nan"))
        n_not_converged = row.get("n_not_converged", float("nan"))
        n_solves = row.get("n_solves", float("nan"))
        logger.info(
            "%-24s %6.4f+-%-6.4f %8.2fx %6d/%-6d %6.1f GiB",
            row["label"],
            row["forward_mean_s"],
            row["forward_std_s"],
            speedup,
            0 if pd.isna(n_not_converged) else int(n_not_converged),
            0 if pd.isna(n_solves) else int(n_solves),
            row["peak_mem_gib"],
        )
    logger.info("=" * 72)

    stalled = summary[summary.get("n_not_converged", pd.Series(dtype=float)) > 0]
    for _, row in stalled.iterrows():
        # Said once more at the end, where it is next to the number it qualifies
        logger.warning(
            "%s: %d of %d solves hit the iteration cap. Its timings are the "
            "cost of exhausting that cap at a residual the other variants beat, "
            "so the speedup against it is a lower bound.",
            row["label"],
            int(row["n_not_converged"]),
            int(row["n_solves"]),
        )


def run_ablation(cfg: DictConfig) -> None:
    assert torch.cuda.is_available(), "The benchmark requires a CUDA-enabled device."

    mhd = is_mhd_env(str(cfg.env_id))
    variants = load_variants(cfg)

    if not mhd:
        logger.info(
            "%s is not an MHD environment: the switches %s have no effect.",
            cfg.env_id,
            ", ".join(MHD_SWITCHES),
        )

    uses_amg = any(
        bool(v.potential_amg) and mhd or v.pressure_amg is None or bool(v.pressure_amg)
        for v in variants
    )
    if uses_amg and not is_pyamg_available():
        raise RuntimeError(
            "The AMG variants need the optional 'pyamg' package for the setup "
            "phase. Install it with `pip install pyamg`, or drop the AMG "
            "variants from `variants`."
        )

    if cfg.get("local_data_path", None):
        fluidgym.config.update(
            "local_data_path",
            str(Path(hydra.utils.get_original_cwd()) / str(cfg.local_data_path)),
        )
    if cfg.get("amg_cache_dir", None):
        os.environ["PHIPICT_AMG_CACHE_DIR"] = str(
            Path(hydra.utils.get_original_cwd()) / str(cfg.amg_cache_dir)
        )

    # The env's registered dt sets the unit of the workload; read it off a
    # throwaway instance so the registered default is used unless overridden
    if cfg.get("dt", None) is not None:
        dt = float(cfg.dt)
    else:
        probe = fluidgym.make(cfg.env_id, **resolve_env_kwargs(cfg))
        dt = float(probe.dt)
        del probe
        free_memory()

    step_length = float(cfg.n_piso_steps) * dt
    logger.info(
        "Ablating %s: dt=%g, step_length=%g (%d PISO steps per env step), "
        "%d warmup + %d measured repeats per variant.",
        cfg.env_id,
        dt,
        step_length,
        int(cfg.n_piso_steps),
        int(cfg.warmup_steps),
        int(cfg.repeats),
    )
    logger.info(
        "Intermediate pressure tolerance of the variants with intermediate_tol: %s. "
        "The final corrector keeps the environment's own tolerance in every variant.",
        cfg.pressure_tol_intermediate
        if isinstance(cfg.pressure_tol_intermediate, str)
        else OmegaConf.to_container(cfg.pressure_tol_intermediate, resolve=True),
    )
    logger.info("Actions: %s.", cfg.actions)

    rows: list[dict[str, Any]] = []
    conv_rows: list[dict[str, Any]] = []

    for variant in variants:
        env_kwargs = resolve_env_kwargs(cfg)
        logger.info(
            "[%s] %s",
            variant.label,
            ", ".join(
                f"{k}={variant[k]}" for k in SWITCHES if mhd or k not in MHD_SWITCHES
            ),
        )

        # Runs before the timed pass: if a variant cannot reach the tolerance,
        # that has to be on the record next to its wall-clock numbers
        if cfg.convergence_check:
            conv_rows += measure_convergence(cfg, variant, env_kwargs, step_length)
            pd.DataFrame(conv_rows).to_csv(cfg.convergence_file, index=False)

        try:
            rows += measure_forward(cfg, variant, env_kwargs, step_length)
        except torch.cuda.OutOfMemoryError:
            logger.warning("[%s] OOM in the forward pass.", variant.label)
            free_memory()
            if not cfg.skip_oom:
                raise
            rows.append(oom_row(variant))

        # Written after every variant so a crashed sweep still leaves usable data
        pd.DataFrame(rows).to_csv(cfg.results_file, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.results_file, index=False)

    conv = pd.DataFrame(conv_rows)
    if not conv.empty:
        conv.to_csv(cfg.convergence_file, index=False)

    summary = attach_convergence(summarize(df, cfg), conv)
    summary.to_csv(cfg.summary_file, index=False)
    log_summary(summary)
    plot_summary(summary, cfg, cfg.plot_file)

    logger.info(
        "Ablation finished. Timings in %s, solver diagnostics in %s, "
        "summary in %s, plot in %s.",
        cfg.results_file,
        cfg.convergence_file,
        cfg.summary_file,
        cfg.plot_file,
    )


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="ablate_solver_improvements",
)
def main(cfg: DictConfig):
    try:
        run_ablation(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


if __name__ == "__main__":
    main()
