"""Ablation of the linear-solver features on fluidgym environments.

Usage:
    python runscripts/benchmark/solver_perf/ablate_solver_perf.py
    python runscripts/benchmark/solver_perf/ablate_solver_perf.py solver_perf_env=hartmann_small_2d
    python runscripts/benchmark/solver_perf/ablate_solver_perf.py variants=[legacy_pict,full] repeats=5
"""

from __future__ import annotations

import gc
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

import fluidgym
from phipict import _C
from phipict.core.piso_simulation import Simulation as PISOSimulation
from phipict.solvers import amg
from phipict.solvers.stats import SolverStatsRecorder

logger = logging.getLogger("solver_perf_ablation")

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_variants(cfg: DictConfig) -> list[DictConfig]:
    """Load the variant configs named in ``cfg.variants``."""
    variants = []
    for name in cfg.variants:
        path = CONFIG_DIR / str(cfg.variant_dir) / f"{name}.yaml"
        if not path.is_file():
            available = sorted(p.stem for p in path.parent.glob("*.yaml"))
            raise FileNotFoundError(f"Unknown variant {name!r}, available: {available}")
        variant = OmegaConf.load(path)
        assert isinstance(variant, DictConfig)
        variants.append(variant)
    return variants


def apply_global_settings(variant: DictConfig) -> None:
    """Process-wide switches: linear solver backend and AMG implementation."""
    _C.SetKrylovSettings(
        enabled=bool(variant.fused_krylov),
        useGraphs=bool(variant.krylov_graphs),
        cgPreconditioner=str(variant.cg_preconditioner),
        bicgPreconditioner=str(variant.bicg_preconditioner),
    )
    amg.USE_NATIVE = bool(variant.native_amg)
    # every variant starts without cached AMG setups (in memory); an on-disk cache
    # (amg_cache_dir) only saves the host-side setup and does not change the solves
    amg.clear_interpolation_cache()


def amg_options(variant: DictConfig) -> dict[str, Any]:
    """Keyword arguments of `amg.hierarchy_for` for this variant."""
    return {
        "reuse_interpolation": bool(variant.share_amg_interpolation),
        "native_dtype": torch.float32 if bool(variant.amg_fp32_vcycle) else None,
    }


def apply_sim_settings(sim: Any, variant: DictConfig) -> None:
    """Per-simulation switches, set on the simulation the environment built."""
    sim.pressure_warm_start = bool(variant.pressure_warm_start)
    sim.pressure_use_amg = variant.pressure_amg if variant.pressure_amg is None else bool(variant.pressure_amg)
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


def make_env(cfg: DictConfig, variant: DictConfig):
    """Build the environment for one variant, with its switches applied."""
    kwargs = OmegaConf.to_container(cfg.env_kwargs, resolve=True)
    assert isinstance(kwargs, dict)
    # reset() builds the simulation and runs its first pressure solve before the
    # variant's switches can be set on it; without this, the automatic pressure AMG
    # would run a host-side AMG setup for variants that do not use it
    auto_min_rows = PISOSimulation.PRESSURE_AMG_AUTO_MIN_ROWS
    if variant.pressure_amg is not None and not bool(variant.pressure_amg):
        PISOSimulation.PRESSURE_AMG_AUTO_MIN_ROWS = 2**62
    try:
        env = fluidgym.make(cfg.env_id, **kwargs, step_length=float(cfg.step_length), differentiable=False)
        env.seed(int(cfg.seed))
        env.reset()
    finally:
        PISOSimulation.PRESSURE_AMG_AUTO_MIN_ROWS = auto_min_rows
    apply_sim_settings(env._sim, variant)
    if cfg.fixed_substeps:
        env._sim.substeps = 1
    return env


def random_actions(env: Any, n: int, seed: int) -> list[torch.Tensor]:
    """The same uniformly random action sequence for every variant.

    Drawn from a dedicated generator rather than the env's RNG, so the sequence does
    not depend on how much randomness the env consumed before.
    """
    space = env.action_space
    low = torch.as_tensor(space.low, dtype=torch.float32)
    high = torch.as_tensor(space.high, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    shape = env._zero_action.shape
    return [
        (low + (high - low) * torch.rand(shape, generator=gen)).to(
            device=env._zero_action.device, dtype=env._zero_action.dtype
        )
        for _ in range(n)
    ]


def run_variant(cfg: DictConfig, variant: DictConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Time the env steps of one variant; statistics come from the warm-up steps."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    apply_global_settings(variant)

    env = make_env(cfg, variant)  # env construction and reset are not timed
    torch.cuda.synchronize()
    sim = env._sim
    actions = random_actions(env, int(cfg.warmup_steps) + int(cfg.repeats), int(cfg.seed))

    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for i in range(int(cfg.warmup_steps) + int(cfg.repeats)):
        warmup = i < int(cfg.warmup_steps)
        piso0 = sim.total_step
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # the recorder synchronises around every solve, so only the untimed warm-up
        # steps record solver statistics
        with SolverStatsRecorder(time_solves=False) if warmup else nullcontext() as rec:
            with torch.no_grad():
                env.step(actions[i])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if warmup and rec is not None:
            stats.update(rec.summary())
        rows.append(
            {
                "variant": str(variant.name),
                "label": str(variant.label),
                "phase": "warmup" if warmup else "measure",
                "step": i,
                "piso_steps": sim.total_step - piso0,
                "step_s": dt,
            }
        )
        logger.info("[%s] step %d%s: %.3f s", variant.name, i, " (warm-up)" if warmup else "", dt)

    vel = env._domain.getBlocks()[0].velocity
    info = {
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "velocity_mean": float(vel.mean()),
        "velocity_absmax": float(vel.abs().max()),
        **{k: v for k, v in stats.items() if k.endswith(("iters_mean", "n_not_converged"))},
    }
    del env, sim
    return rows, info


def summarize(df: pd.DataFrame, infos: dict[str, dict[str, Any]], cfg: DictConfig) -> pd.DataFrame:
    """One row per variant with the mean step time and the speedup over the baseline."""
    rows = []
    for name, group in df[df["phase"] == "measure"].groupby("variant", sort=False):
        rows.append(
            {
                "variant": name,
                "label": group["label"].iloc[0],
                "piso_steps_per_env_step": float(group["piso_steps"].mean()),
                "step_mean_s": float(group["step_s"].mean()),
                "step_std_s": float(group["step_s"].std(ddof=1)) if len(group) > 1 else 0.0,
                **infos[name],
            }
        )
    summary = pd.DataFrame(rows)
    base = summary[summary["variant"] == str(cfg.baseline_variant)]
    if not base.empty:
        summary["speedup_vs_baseline"] = float(base["step_mean_s"].iloc[0]) / summary["step_mean_s"]
    ref = summary[summary["variant"] == str(cfg.reference_variant)]
    if not ref.empty:
        summary["slowdown_vs_reference"] = summary["step_mean_s"] / float(ref["step_mean_s"].iloc[0])
    return summary


@hydra.main(version_base="1.3", config_path="../../configs", config_name="solver_perf_ablation")
def main(cfg: DictConfig) -> None:
    if cfg.local_data_path:
        fluidgym.config.update("local_data_path", str(cfg.local_data_path))
    if cfg.amg_cache_dir:
        os.environ["PHIPICT_AMG_CACHE_DIR"] = str(cfg.amg_cache_dir)

    variants = load_variants(cfg)
    all_rows: list[dict[str, Any]] = []
    infos: dict[str, dict[str, Any]] = {}
    for variant in variants:
        logger.info("Variant %s (%s)", variant.name, variant.label)
        rows, info = run_variant(cfg, variant)
        all_rows.extend(rows)
        infos[str(variant.name)] = info
        pd.DataFrame(all_rows).to_csv(cfg.results_file, index=False)

    summary = summarize(pd.DataFrame(all_rows), infos, cfg)
    summary.to_csv(cfg.summary_file, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        logger.info("Summary:\n%s", summary)

    # restore the defaults of the process-wide switches
    _C.SetKrylovSettings()
    amg.USE_NATIVE = True


if __name__ == "__main__":
    main()
