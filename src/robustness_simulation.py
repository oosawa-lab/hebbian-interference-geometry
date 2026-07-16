#!/usr/bin/env python3
"""
Reviewer-resistance simulation script for L4/V04.

Purpose
-------
This script is not meant to replace the main L4/V04 simulation. It is a
stress-test suite for possible reviewer concerns.

It tests whether the main qualitative result,

    correlated memory patterns induce a positive intensive excess
    Hebbian-writing cost,

is robust to:

  1. different correlated-pattern generators,
  2. different correlation strengths,
  3. different interference strengths lambda,
  4. different system sizes N,
  5. nearest-load finite-size matching,
  6. bootstrap confidence intervals and sign robustness,
  7. Randers writing/erasing asymmetry under multiple one-form definitions.

The script writes CSV summaries and diagnostic figures. Use the output to
support V05 or a supplementary robustness appendix if needed.

Safe interpretation
-------------------
A robust positive finite-size Delta F_int supports the manuscript's finite-size
statistical-physics claim. This script does not prove a nonzero thermodynamic
limit unless the finite-size scaling results clearly support it.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class TrialRow:
    N: int
    load: float
    actual_load: float
    P: int
    generator: str
    corr: float
    lambda_interference: float
    randers_alpha: float
    randers_mode: str
    rep: int
    cost_uncorr_int: float
    cost_corr_int: float
    deltaF_int: float
    cost_uncorr_cum: float
    cost_corr_cum: float
    deltaF_cum: float
    randers_uncorr_asym: float
    randers_corr_asym: float
    delta_randers_asym: float
    mean_pair_overlap_uncorr: float
    mean_pair_overlap_corr: float


def ensure_dirs(outdir: Path) -> Tuple[Path, Path]:
    data_dir = outdir / "data"
    figs_dir = outdir / "figs"
    data_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, figs_dir


def spin_patterns_uncorrelated(rng: np.random.Generator, P: int, N: int) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(P, N))


def spin_patterns_latent_template(rng: np.random.Generator, P: int, N: int, corr: float) -> np.ndarray:
    """Common-template correlation model used in the main V04 simulations."""
    corr = float(np.clip(corr, -0.999, 0.999))
    template = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    flip_prob = (1.0 - corr) / 2.0
    flips = rng.random((P, N)) < flip_prob
    patterns = np.tile(template, (P, 1)).astype(np.int8)
    patterns[flips] *= -1
    return patterns


def spin_patterns_block_latent(rng: np.random.Generator, P: int, N: int, corr: float, n_blocks: int = 4) -> np.ndarray:
    """
    Blockwise latent-template model.

    Each block has its own latent template. This reduces the risk that the
    result is an artifact of a single global template direction.
    """
    corr = float(np.clip(corr, -0.999, 0.999))
    n_blocks = max(1, min(n_blocks, N))
    block_edges = np.linspace(0, N, n_blocks + 1, dtype=int)
    patterns = np.empty((P, N), dtype=np.int8)
    for b in range(n_blocks):
        lo, hi = block_edges[b], block_edges[b + 1]
        width = hi - lo
        template = rng.choice(np.array([-1, 1], dtype=np.int8), size=width)
        flip_prob = (1.0 - corr) / 2.0
        flips = rng.random((P, width)) < flip_prob
        block = np.tile(template, (P, 1)).astype(np.int8)
        block[flips] *= -1
        patterns[:, lo:hi] = block
    return patterns


def spin_patterns_mixture_templates(rng: np.random.Generator, P: int, N: int, corr: float, n_templates: int = 3) -> np.ndarray:
    """
    Mixture-of-templates correlation model.

    Patterns are assigned to one of several latent clusters. This creates
    clustered memory correlations rather than a single global correlation.
    """
    corr = float(np.clip(corr, -0.999, 0.999))
    n_templates = max(1, n_templates)
    templates = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_templates, N))
    assignments = rng.integers(0, n_templates, size=P)
    flip_prob = (1.0 - corr) / 2.0
    patterns = templates[assignments].copy()
    flips = rng.random((P, N)) < flip_prob
    patterns[flips] *= -1
    return patterns.astype(np.int8)


def spin_patterns_biased_independent(rng: np.random.Generator, P: int, N: int, corr: float) -> np.ndarray:
    """
    Biased independent pattern model.

    This is not pairwise template correlation in the same sense. It tests
    whether a positive excess cost also appears when correlations are induced
    through biased marginals. corr controls the spin bias m approximately.
    """
    m = float(np.clip(corr, 0.0, 0.95))
    prob_plus = (1.0 + m) / 2.0
    return np.where(rng.random((P, N)) < prob_plus, 1, -1).astype(np.int8)


PATTERN_GENERATORS: Dict[str, Callable[[np.random.Generator, int, int, float], np.ndarray]] = {
    "latent": spin_patterns_latent_template,
    "block": spin_patterns_block_latent,
    "mixture": spin_patterns_mixture_templates,
    "biased": spin_patterns_biased_independent,
}


def hebbian_direction(pattern: np.ndarray) -> np.ndarray:
    N = pattern.shape[0]
    iu = np.triu_indices(N, k=1)
    raw = np.outer(pattern, pattern).astype(np.float64)[iu]
    raw /= math.sqrt(raw.size)
    return raw


def directions_from_patterns(patterns: np.ndarray) -> np.ndarray:
    return np.vstack([hebbian_direction(patterns[p]) for p in range(patterns.shape[0])])


def pair_overlap_stat(patterns: np.ndarray) -> float:
    """Mean absolute pair overlap |N^{-1} xi_mu dot xi_nu| over mu<nu."""
    P, N = patterns.shape
    if P < 2:
        return 0.0
    X = patterns.astype(float)
    G = (X @ X.T) / N
    iu = np.triu_indices(P, k=1)
    return float(np.mean(np.abs(G[iu])))


def cost_against_memory(stored_dirs: np.ndarray, query_dir: np.ndarray, lam: float) -> Tuple[float, float]:
    base = float(np.dot(query_dir, query_dir))
    if stored_dirs.shape[0] == 0:
        return math.sqrt(base), math.sqrt(base)
    overlaps = stored_dirs @ query_dir
    sq = overlaps * overlaps
    cum = base + lam * float(np.sum(sq))
    intensive = base + lam * float(np.mean(sq))
    return math.sqrt(max(cum, 0.0)), math.sqrt(max(intensive, 0.0))


def randers_one_form(stored_dirs: np.ndarray, query_dir: np.ndarray, alpha: float, mode: str, rng: np.random.Generator) -> float:
    """
    Randers one-form A(V).

    mode='mean':
        A aligned with empirical mean memory direction.

    mode='pc1':
        A aligned with the first principal direction of stored memory directions.

    mode='random-fixed':
        A aligned with a random normalized direction. This is a negative-control
        style comparison; it should not be overinterpreted as biology.
    """
    if stored_dirs.shape[0] == 0 or alpha == 0.0:
        return 0.0

    if mode == "mean":
        a = np.mean(stored_dirs, axis=0)
    elif mode == "pc1":
        centered = stored_dirs - np.mean(stored_dirs, axis=0, keepdims=True)
        # Use SVD on the smaller representation. This is acceptable for the
        # intended sizes. Fall back to mean if SVD fails.
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            a = vh[0]
        except np.linalg.LinAlgError:
            a = np.mean(stored_dirs, axis=0)
    elif mode == "random-fixed":
        a = rng.normal(size=query_dir.shape[0])
    else:
        raise ValueError(f"Unknown randers_mode: {mode}")

    norm = float(np.linalg.norm(a))
    if norm < 1e-14:
        return 0.0
    return float(alpha * np.dot(a / norm, query_dir))


def one_trial(
    rng: np.random.Generator,
    N: int,
    load: float,
    generator: str,
    corr: float,
    lam: float,
    alpha: float,
    randers_mode: str,
    rep: int,
) -> TrialRow:
    P = max(1, int(round(N * load)))
    actual_load = P / N

    # matched uncorrelated null
    uncorr_patterns = spin_patterns_uncorrelated(rng, P + 1, N)
    corr_patterns = PATTERN_GENERATORS[generator](rng, P + 1, N, corr)

    uncorr_dirs = directions_from_patterns(uncorr_patterns)
    corr_dirs = directions_from_patterns(corr_patterns)

    uncorr_stored, uncorr_query = uncorr_dirs[:P], uncorr_dirs[P]
    corr_stored, corr_query = corr_dirs[:P], corr_dirs[P]

    u_cum, u_int = cost_against_memory(uncorr_stored, uncorr_query, lam)
    c_cum, c_int = cost_against_memory(corr_stored, corr_query, lam)

    Au = randers_one_form(uncorr_stored, uncorr_query, alpha, randers_mode, rng)
    Ac = randers_one_form(corr_stored, corr_query, alpha, randers_mode, rng)

    return TrialRow(
        N=N,
        load=float(load),
        actual_load=float(actual_load),
        P=P,
        generator=generator,
        corr=float(corr),
        lambda_interference=float(lam),
        randers_alpha=float(alpha),
        randers_mode=randers_mode,
        rep=rep,
        cost_uncorr_int=float(u_int),
        cost_corr_int=float(c_int),
        deltaF_int=float(c_int - u_int),
        cost_uncorr_cum=float(u_cum),
        cost_corr_cum=float(c_cum),
        deltaF_cum=float(c_cum - u_cum),
        randers_uncorr_asym=float(2.0 * Au),
        randers_corr_asym=float(2.0 * Ac),
        delta_randers_asym=float(2.0 * (Ac - Au)),
        mean_pair_overlap_uncorr=pair_overlap_stat(uncorr_patterns[:P]),
        mean_pair_overlap_corr=pair_overlap_stat(corr_patterns[:P]),
    )


def write_trial_csv(rows: List[TrialRow], path: Path) -> None:
    fields = list(TrialRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def mean_sem_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float, float, float]:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    frac_pos = float(np.mean(values > 0.0))
    if n_boot <= 0 or n == 1:
        return mean, sem, float("nan"), float("nan"), frac_pos
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot[b] = np.mean(sample)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, sem, float(lo), float(hi), frac_pos


def summarize(rows: List[TrialRow], n_boot: int, seed: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed + 999)
    groups: Dict[Tuple[int, float, str, float, float, float, str], List[TrialRow]] = {}
    for r in rows:
        key = (r.N, r.load, r.generator, r.corr, r.lambda_interference, r.randers_alpha, r.randers_mode)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, float]] = []
    for key, rs in sorted(groups.items()):
        N, load, generator, corr, lam, alpha, randers_mode = key
        actual_load = float(np.mean([r.actual_load for r in rs]))
        P = int(round(np.mean([r.P for r in rs])))
        row: Dict[str, float] = {
            "N": N,
            "nominal_load": load,
            "actual_load": actual_load,
            "P": P,
            "generator": generator,
            "corr": corr,
            "lambda_interference": lam,
            "randers_alpha": alpha,
            "randers_mode": randers_mode,
            "n_reps": len(rs),
        }
        for name in [
            "deltaF_int",
            "deltaF_cum",
            "delta_randers_asym",
            "mean_pair_overlap_uncorr",
            "mean_pair_overlap_corr",
            "cost_uncorr_int",
            "cost_corr_int",
            "cost_uncorr_cum",
            "cost_corr_cum",
            "randers_uncorr_asym",
            "randers_corr_asym",
        ]:
            arr = np.array([getattr(r, name) for r in rs], dtype=float)
            m, sem, lo, hi, frac_pos = mean_sem_ci(arr, n_boot, rng)
            row[f"{name}_mean"] = m
            row[f"{name}_sem"] = sem
            row[f"{name}_ci95_lo"] = lo
            row[f"{name}_ci95_hi"] = hi
            row[f"{name}_frac_positive"] = frac_pos
        out.append(row)
    return out


def write_dict_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def nearest_load_rows(summary_rows: List[Dict[str, float]], reference_loads: List[float], tolerance: float) -> List[Dict[str, float]]:
    # group by non-load parameters and N
    buckets: Dict[Tuple[str, float, float, float, str, int], List[Dict[str, float]]] = {}
    for r in summary_rows:
        key = (
            str(r["generator"]),
            float(r["corr"]),
            float(r["lambda_interference"]),
            float(r["randers_alpha"]),
            str(r["randers_mode"]),
            int(r["N"]),
        )
        buckets.setdefault(key, []).append(r)

    out: List[Dict[str, float]] = []
    # For each parameter set except N, each ref load, take nearest row for each N.
    param_keys = sorted({k[:-1] for k in buckets.keys()})
    for pkey in param_keys:
        for ref in reference_loads:
            Ns = sorted([k[-1] for k in buckets.keys() if k[:-1] == pkey])
            for N in Ns:
                rows = buckets[pkey + (N,)]
                best = min(rows, key=lambda r: abs(float(r["actual_load"]) - ref))
                dist = abs(float(best["actual_load"]) - ref)
                if tolerance >= 0 and dist > tolerance:
                    continue
                rec = dict(best)
                rec["reference_load"] = float(ref)
                rec["load_distance"] = float(dist)
                out.append(rec)
    return out


def finite_size_power_fit(scaling_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Fit DeltaF_int ≈ a N^{-beta} for positive means, by linear regression in log-log.

    This is diagnostic only. It should not be overinterpreted when point counts are small.
    """
    groups: Dict[Tuple[str, float, float, str, float], List[Dict[str, float]]] = {}
    for r in scaling_rows:
        key = (
            str(r["generator"]),
            float(r["corr"]),
            float(r["lambda_interference"]),
            str(r["randers_mode"]),
            float(r["reference_load"]),
        )
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, float]] = []
    for key, rows in sorted(groups.items()):
        generator, corr, lam, randers_mode, ref_load = key
        rows = sorted(rows, key=lambda r: int(r["N"]))
        xs, ys = [], []
        for r in rows:
            y = float(r["deltaF_int_mean"])
            if y > 0:
                xs.append(float(r["N"]))
                ys.append(y)
        if len(xs) >= 3:
            logx = np.log(np.array(xs))
            logy = np.log(np.array(ys))
            slope, intercept = np.polyfit(logx, logy, 1)
            pred = intercept + slope * logx
            ss_res = float(np.sum((logy - pred) ** 2))
            ss_tot = float(np.sum((logy - np.mean(logy)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            beta = -float(slope)
            a = float(math.exp(intercept))
        else:
            beta, a, r2 = float("nan"), float("nan"), float("nan")
        out.append({
            "generator": generator,
            "corr": corr,
            "lambda_interference": lam,
            "randers_mode": randers_mode,
            "reference_load": ref_load,
            "n_points": len(xs),
            "prefactor_a": a,
            "decay_beta": beta,
            "loglog_r2": r2,
        })
    return out


def plot_corr_sweep(summary: List[Dict[str, float]], figs_dir: Path, target_N: int, target_load: float, lam: float, generator: str) -> None:
    rows = [
        r for r in summary
        if int(r["N"]) == target_N
        and str(r["generator"]) == generator
        and abs(float(r["lambda_interference"]) - lam) < 1e-12
        and abs(float(r["actual_load"]) - target_load) < 0.03
        and str(r["randers_mode"]) == "mean"
    ]
    if not rows:
        return
    # For each corr take closest load row.
    by_corr: Dict[float, List[Dict[str, float]]] = {}
    for r in rows:
        by_corr.setdefault(float(r["corr"]), []).append(r)
    chosen = []
    for corr, rs in by_corr.items():
        chosen.append(min(rs, key=lambda r: abs(float(r["actual_load"]) - target_load)))
    chosen = sorted(chosen, key=lambda r: float(r["corr"]))

    x = np.array([r["corr"] for r in chosen], dtype=float)
    y = np.array([r["deltaF_int_mean"] for r in chosen], dtype=float)
    lo = np.array([r["deltaF_int_ci95_lo"] for r in chosen], dtype=float)
    hi = np.array([r["deltaF_int_ci95_hi"] for r in chosen], dtype=float)

    plt.figure(figsize=(7.0, 4.6))
    plt.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", linewidth=1.5, capsize=3)
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel(r"Correlation parameter")
    plt.ylabel(r"Intensive excess cost $\Delta F_{\mathrm{int}}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / f"robust_R1_corr_sweep_{generator}_N{target_N}.png", dpi=240)
    plt.savefig(figs_dir / f"robust_R1_corr_sweep_{generator}_N{target_N}.pdf")
    plt.close()


def plot_generator_comparison(summary: List[Dict[str, float]], figs_dir: Path, target_N: int, target_load: float, corr: float, lam: float) -> None:
    chosen = []
    for gen in sorted({str(r["generator"]) for r in summary}):
        rows = [
            r for r in summary
            if int(r["N"]) == target_N
            and str(r["generator"]) == gen
            and abs(float(r["corr"]) - corr) < 1e-12
            and abs(float(r["lambda_interference"]) - lam) < 1e-12
            and str(r["randers_mode"]) == "mean"
        ]
        if rows:
            chosen.append(min(rows, key=lambda r: abs(float(r["actual_load"]) - target_load)))
    if not chosen:
        return
    x = np.arange(len(chosen))
    labels = [str(r["generator"]) for r in chosen]
    y = np.array([r["deltaF_int_mean"] for r in chosen], dtype=float)
    lo = np.array([r["deltaF_int_ci95_lo"] for r in chosen], dtype=float)
    hi = np.array([r["deltaF_int_ci95_hi"] for r in chosen], dtype=float)

    plt.figure(figsize=(7.0, 4.6))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", capsize=3)
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(x, labels)
    plt.xlabel("Correlated-pattern generator")
    plt.ylabel(r"Intensive excess cost $\Delta F_{\mathrm{int}}$")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(figs_dir / f"robust_R2_generator_comparison_N{target_N}.png", dpi=240)
    plt.savefig(figs_dir / f"robust_R2_generator_comparison_N{target_N}.pdf")
    plt.close()


def plot_lambda_sweep(summary: List[Dict[str, float]], figs_dir: Path, target_N: int, target_load: float, corr: float, generator: str) -> None:
    rows = [
        r for r in summary
        if int(r["N"]) == target_N
        and str(r["generator"]) == generator
        and abs(float(r["corr"]) - corr) < 1e-12
        and str(r["randers_mode"]) == "mean"
        and abs(float(r["actual_load"]) - target_load) < 0.03
    ]
    if not rows:
        return
    by_lam: Dict[float, List[Dict[str, float]]] = {}
    for r in rows:
        by_lam.setdefault(float(r["lambda_interference"]), []).append(r)
    chosen = []
    for lam, rs in by_lam.items():
        chosen.append(min(rs, key=lambda r: abs(float(r["actual_load"]) - target_load)))
    chosen = sorted(chosen, key=lambda r: float(r["lambda_interference"]))
    x = np.array([r["lambda_interference"] for r in chosen], dtype=float)
    y = np.array([r["deltaF_int_mean"] for r in chosen], dtype=float)
    lo = np.array([r["deltaF_int_ci95_lo"] for r in chosen], dtype=float)
    hi = np.array([r["deltaF_int_ci95_hi"] for r in chosen], dtype=float)

    plt.figure(figsize=(7.0, 4.6))
    plt.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", linewidth=1.5, capsize=3)
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel(r"Interference strength $\lambda$")
    plt.ylabel(r"Intensive excess cost $\Delta F_{\mathrm{int}}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / f"robust_R3_lambda_sweep_{generator}_N{target_N}.png", dpi=240)
    plt.savefig(figs_dir / f"robust_R3_lambda_sweep_{generator}_N{target_N}.pdf")
    plt.close()


def plot_finite_size(scaling: List[Dict[str, float]], figs_dir: Path, generator: str, corr: float, lam: float, ref_load: float) -> None:
    rows = [
        r for r in scaling
        if str(r["generator"]) == generator
        and abs(float(r["corr"]) - corr) < 1e-12
        and abs(float(r["lambda_interference"]) - lam) < 1e-12
        and abs(float(r["reference_load"]) - ref_load) < 1e-12
        and str(r["randers_mode"]) == "mean"
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(r["N"]))
    x = np.array([r["N"] for r in rows], dtype=float)
    y = np.array([r["deltaF_int_mean"] for r in rows], dtype=float)
    lo = np.array([r["deltaF_int_ci95_lo"] for r in rows], dtype=float)
    hi = np.array([r["deltaF_int_ci95_hi"] for r in rows], dtype=float)

    plt.figure(figsize=(7.0, 4.6))
    plt.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", linewidth=1.5, capsize=3)
    plt.xscale("log", base=2)
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel(r"System size $N$")
    plt.ylabel(r"Intensive excess cost $\Delta F_{\mathrm{int}}$")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(figs_dir / f"robust_R4_finite_size_{generator}_corr{corr:g}_load{ref_load:g}.png", dpi=240)
    plt.savefig(figs_dir / f"robust_R4_finite_size_{generator}_corr{corr:g}_load{ref_load:g}.pdf")
    plt.close()


def plot_randers_modes(summary: List[Dict[str, float]], figs_dir: Path, target_N: int, target_load: float, corr: float, lam: float, generator: str) -> None:
    chosen = []
    for mode in sorted({str(r["randers_mode"]) for r in summary}):
        rows = [
            r for r in summary
            if int(r["N"]) == target_N
            and str(r["generator"]) == generator
            and abs(float(r["corr"]) - corr) < 1e-12
            and abs(float(r["lambda_interference"]) - lam) < 1e-12
            and str(r["randers_mode"]) == mode
        ]
        if rows:
            chosen.append(min(rows, key=lambda r: abs(float(r["actual_load"]) - target_load)))
    if not chosen:
        return
    x = np.arange(len(chosen))
    labels = [str(r["randers_mode"]) for r in chosen]
    y = np.array([r["delta_randers_asym_mean"] for r in chosen], dtype=float)
    lo = np.array([r["delta_randers_asym_ci95_lo"] for r in chosen], dtype=float)
    hi = np.array([r["delta_randers_asym_ci95_hi"] for r in chosen], dtype=float)

    plt.figure(figsize=(7.0, 4.6))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", capsize=3)
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(x, labels)
    plt.xlabel("Randers one-form mode")
    plt.ylabel(r"Excess asymmetry $\Delta[F_R(V)-F_R(-V)]$")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(figs_dir / f"robust_R5_randers_mode_comparison_N{target_N}.png", dpi=240)
    plt.savefig(figs_dir / f"robust_R5_randers_mode_comparison_N{target_N}.pdf")
    plt.close()


def write_readme(outdir: Path, args: argparse.Namespace, n_trials: int) -> None:
    text = f"""L4/V04 reviewer-resistance robustness output
============================================

This directory was generated by:

  L4_V04_reviewer_robustness_simulation.py

Purpose
-------
These calculations stress-test the L4/V04 manuscript against reviewer concerns.

They test robustness across:
- pattern generators: {args.generators}
- correlation parameters: {args.corrs}
- interference strengths: {args.lambdas}
- system sizes: {args.Ns}
- memory loads: {args.loads}
- Randers one-form modes: {args.randers_modes}

Total trial rows:
  {n_trials}

Main CSV files
--------------
data/L4_V04_robustness_trial_rows.csv
data/L4_V04_robustness_summary.csv
data/L4_V04_robustness_nearest_load_scaling.csv
data/L4_V04_robustness_finite_size_power_fits.csv

Main figures
------------
figs/robust_R1_corr_sweep_*.png
figs/robust_R2_generator_comparison_*.png
figs/robust_R3_lambda_sweep_*.png
figs/robust_R4_finite_size_*.png
figs/robust_R5_randers_mode_comparison_*.png

How to use these results
------------------------
Use these results as a robustness appendix or supplementary analysis.

Safe statement:
  The qualitative finite-size excess writing cost is robust across several
  correlated-pattern generators and parameter choices.

Avoid overclaiming:
  Do not claim a nonzero thermodynamic-limit excess unless the finite-size
  power-fit results clearly support it.
"""
    (outdir / "README_L4_V04_reviewer_robustness_output.txt").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reviewer-resistance robustness simulation for L4/V04.")
    p.add_argument("--outdir", type=str, default="L4_V04_reviewer_robustness_results")
    p.add_argument("--Ns", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--loads", type=float, nargs="+", default=[0.10, 0.30, 0.60])
    p.add_argument("--generators", type=str, nargs="+", default=["latent", "block", "mixture", "biased"])
    p.add_argument("--corrs", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--randers-alpha", type=float, default=0.15)
    p.add_argument("--randers-modes", type=str, nargs="+", default=["mean", "pc1", "random-fixed"])
    p.add_argument("--reps", type=int, default=80)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--reference-loads", type=float, nargs="+", default=[0.10, 0.30, 0.60])
    p.add_argument("--nearest-load-tolerance", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.outdir = "L4_V04_reviewer_robustness_quick"
        args.Ns = [32, 64]
        args.loads = [0.10, 0.60]
        args.generators = ["latent", "mixture"]
        args.corrs = [0.25, 0.35]
        args.lambdas = [1.0, 2.0]
        args.randers_modes = ["mean", "pc1"]
        args.reps = 10
        args.bootstrap = 50
        args.reference_loads = [0.10, 0.60]
        args.nearest_load_tolerance = 0.05

    outdir = Path(args.outdir)
    data_dir, figs_dir = ensure_dirs(outdir)
    rng = np.random.default_rng(args.seed)

    # Validate generators and modes.
    for g in args.generators:
        if g not in PATTERN_GENERATORS:
            raise ValueError(f"Unknown generator {g}. Available: {sorted(PATTERN_GENERATORS)}")
    for mode in args.randers_modes:
        if mode not in ["mean", "pc1", "random-fixed"]:
            raise ValueError(f"Unknown randers mode {mode}")

    rows: List[TrialRow] = []
    total = (
        len(args.Ns) * len(args.loads) * len(args.generators) * len(args.corrs)
        * len(args.lambdas) * len(args.randers_modes) * args.reps
    )
    counter = 0
    for N in args.Ns:
        for load in args.loads:
            for generator in args.generators:
                for corr in args.corrs:
                    for lam in args.lambdas:
                        for mode in args.randers_modes:
                            for rep in range(args.reps):
                                rows.append(one_trial(
                                    rng=rng,
                                    N=N,
                                    load=load,
                                    generator=generator,
                                    corr=corr,
                                    lam=lam,
                                    alpha=args.randers_alpha,
                                    randers_mode=mode,
                                    rep=rep,
                                ))
                                counter += 1
        print(f"Completed N={N}; progress {counter}/{total}")

    write_trial_csv(rows, data_dir / "L4_V04_robustness_trial_rows.csv")

    summary = summarize(rows, args.bootstrap, args.seed)
    write_dict_csv(summary, data_dir / "L4_V04_robustness_summary.csv")

    scaling = nearest_load_rows(summary, args.reference_loads, args.nearest_load_tolerance)
    write_dict_csv(scaling, data_dir / "L4_V04_robustness_nearest_load_scaling.csv")

    fits = finite_size_power_fit(scaling)
    write_dict_csv(fits, data_dir / "L4_V04_robustness_finite_size_power_fits.csv")

    # Diagnostic figures with defaults centered around the main manuscript parameters.
    target_N = max(args.Ns)
    target_load = 0.60 if 0.60 in args.loads else args.loads[-1]
    target_corr = 0.35 if 0.35 in args.corrs else args.corrs[len(args.corrs)//2]
    target_lam = 2.0 if 2.0 in args.lambdas else args.lambdas[len(args.lambdas)//2]
    target_gen = "latent" if "latent" in args.generators else args.generators[0]

    for gen in args.generators:
        plot_corr_sweep(summary, figs_dir, target_N, target_load, target_lam, gen)

    plot_generator_comparison(summary, figs_dir, target_N, target_load, target_corr, target_lam)

    for gen in args.generators:
        plot_lambda_sweep(summary, figs_dir, target_N, target_load, target_corr, gen)

    for gen in args.generators:
        plot_finite_size(scaling, figs_dir, gen, target_corr, target_lam, target_load)

    plot_randers_modes(summary, figs_dir, target_N, target_load, target_corr, target_lam, target_gen)

    write_readme(outdir, args, len(rows))

    print("Reviewer-resistance robustness simulation complete.")
    print(f"Output directory: {outdir.resolve()}")
    print("Recommended files to inspect first:")
    print(f"  {data_dir / 'L4_V04_robustness_summary.csv'}")
    print(f"  {data_dir / 'L4_V04_robustness_finite_size_power_fits.csv'}")
    print(f"  {figs_dir / ('robust_R2_generator_comparison_N' + str(target_N) + '.png')}")


if __name__ == "__main__":
    main()
