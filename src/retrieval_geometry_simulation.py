#!/usr/bin/env python3
"""
L4 special-issue simulation:
Hebbian interference geometry versus associative-memory retrieval.

Main objectives
---------------
1. Test whether the geometric Hebbian writing cost predicts retrieval damage.
2. Compare correlated memories with matched uncorrelated controls.
3. Benchmark simulated writing costs against an overlap-moment prediction.
4. Produce finite-size, cue-noise, ensemble, and correlation-strength summaries.

The script is designed for local execution. It:
- uses only NumPy and Matplotlib,
- saves PNG figures only,
- writes trial-level and summary CSV files,
- supports checkpointing and resume,
- avoids constructing huge Hebbian direction vectors by using the exact identity

    <Vhat^mu,Vhat^nu> = (N m_{mu nu}^2 - 1)/(N-1).

Python 3.10 or later is recommended.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class TrialRow:
    N: int
    nominal_load: float
    actual_load: float
    P: int
    generator: str
    corr: float
    condition: str
    rep: int
    cue_noise: float
    cue_id: int
    target_index: int
    cost_int: float
    cost_cum: float
    initial_overlap: float
    final_overlap: float
    retrieval_error: float
    success: int
    iterations: int


@dataclass
class RealizationRow:
    N: int
    nominal_load: float
    actual_load: float
    P: int
    generator: str
    corr: float
    condition: str
    rep: int
    mean_pair_overlap_abs: float
    mean_q2: float
    mean_cost_int: float
    mean_cost_cum: float


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dirs(outdir: Path) -> Tuple[Path, Path]:
    data_dir = outdir / "data"
    figs_dir = outdir / "figs"
    data_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, figs_dir


def append_dataclass_rows(path: Path, rows: Sequence[object], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_dict_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Some grouped summaries contain optional grouping columns.
    # Use the union of all keys while preserving first appearance order.
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_sem(values: Sequence[float]) -> Tuple[float, float, int]:
    a = np.asarray(values, dtype=float)
    n = int(a.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(a))
    sem = float(np.std(a, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean, sem, n


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties; SciPy-free."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_vals = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearsonr(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den < 1e-15:
        return float("nan")
    return float(np.dot(x, y) / den)


def spearmanr(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    return pearsonr(rankdata(x), rankdata(y))


def nearest(value: float, candidates: Sequence[float]) -> float:
    return min(candidates, key=lambda z: abs(float(z) - float(value)))


# ---------------------------------------------------------------------
# Pattern ensembles
# ---------------------------------------------------------------------

def patterns_uncorrelated(rng: np.random.Generator, P: int, N: int, corr: float = 0.0) -> np.ndarray:
    del corr
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(P, N))


def patterns_latent(rng: np.random.Generator, P: int, N: int, corr: float) -> np.ndarray:
    """
    Global latent-template ensemble.

    Each pattern is obtained from one common random template through
    independent bit flips. The single-pattern alignment with the template
    is approximately corr; pair overlap is approximately corr^2.
    """
    corr = float(np.clip(corr, 0.0, 0.999))
    template = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    flips = rng.random((P, N)) < (1.0 - corr) / 2.0
    X = np.tile(template, (P, 1)).astype(np.int8)
    X[flips] *= -1
    return X


def patterns_block(rng: np.random.Generator, P: int, N: int, corr: float, n_blocks: int = 4) -> np.ndarray:
    """
    Block-latent ensemble with independent templates in several spatial blocks.
    """
    corr = float(np.clip(corr, 0.0, 0.999))
    edges = np.linspace(0, N, max(1, n_blocks) + 1, dtype=int)
    X = np.empty((P, N), dtype=np.int8)
    for lo, hi in zip(edges[:-1], edges[1:]):
        width = hi - lo
        template = rng.choice(np.array([-1, 1], dtype=np.int8), size=width)
        flips = rng.random((P, width)) < (1.0 - corr) / 2.0
        block = np.tile(template, (P, 1)).astype(np.int8)
        block[flips] *= -1
        X[:, lo:hi] = block
    return X


def patterns_mixture(rng: np.random.Generator, P: int, N: int, corr: float, n_templates: int = 3) -> np.ndarray:
    """
    Mixture-of-templates ensemble. Patterns form several memory clusters.
    """
    corr = float(np.clip(corr, 0.0, 0.999))
    K = max(1, int(n_templates))
    templates = rng.choice(np.array([-1, 1], dtype=np.int8), size=(K, N))
    labels = rng.integers(0, K, size=P)
    X = templates[labels].copy()
    flips = rng.random((P, N)) < (1.0 - corr) / 2.0
    X[flips] *= -1
    return X.astype(np.int8)


GENERATORS = {
    "latent": patterns_latent,
    "block": patterns_block,
    "mixture": patterns_mixture,
}


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def pattern_overlap_matrix(X: np.ndarray) -> np.ndarray:
    N = X.shape[1]
    return (X.astype(np.float64) @ X.astype(np.float64).T) / float(N)


def hebbian_direction_overlap_matrix(X: np.ndarray) -> np.ndarray:
    """
    Exact off-diagonal overlap matrix of normalized zero-diagonal
    Hebbian directions.
    """
    N = X.shape[1]
    M = pattern_overlap_matrix(X)
    Q = (N * M * M - 1.0) / float(N - 1)
    np.fill_diagonal(Q, 1.0)
    return Q


def geometric_costs(X: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Per-pattern intensive and cumulative costs against all other memories.
    """
    P = X.shape[0]
    Q = hebbian_direction_overlap_matrix(X)
    if P <= 1:
        cost_int = np.ones(P)
        cost_cum = np.ones(P)
        return cost_int, cost_cum, 0.0, 0.0

    Q2 = Q * Q
    np.fill_diagonal(Q2, 0.0)
    sum_q2 = np.sum(Q2, axis=1)
    mean_q2 = sum_q2 / float(P - 1)

    cost_int = np.sqrt(1.0 + lam * mean_q2)
    cost_cum = np.sqrt(1.0 + lam * sum_q2)

    M = pattern_overlap_matrix(X)
    iu = np.triu_indices(P, k=1)
    mean_pair_overlap_abs = float(np.mean(np.abs(M[iu]))) if iu[0].size else 0.0
    overall_mean_q2 = float(np.mean(Q2[iu])) if iu[0].size else 0.0
    return cost_int, cost_cum, mean_pair_overlap_abs, overall_mean_q2


# ---------------------------------------------------------------------
# Hopfield retrieval
# ---------------------------------------------------------------------

def make_cues(
    rng: np.random.Generator,
    X: np.ndarray,
    cue_noise: float,
    n_cues: int,
) -> Tuple[np.ndarray, np.ndarray]:
    P, N = X.shape
    targets = rng.integers(0, P, size=n_cues)
    S = X[targets].copy()
    flips = rng.random((n_cues, N)) < cue_noise
    S[flips] *= -1
    return S.astype(np.int8), targets.astype(int)


def hopfield_recall_batch(
    X: np.ndarray,
    cues: np.ndarray,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched synchronous Hopfield recall without explicitly constructing W.

    W = X^T X / N with zero diagonal.
    Therefore:
        S W = (S X^T) X / N - (P/N) S.
    """
    P, N = X.shape
    Xf = X.astype(np.float64)
    S = cues.astype(np.float64).copy()
    iterations = np.zeros(S.shape[0], dtype=int)
    active = np.ones(S.shape[0], dtype=bool)

    for step in range(1, max_steps + 1):
        h = ((S @ Xf.T) @ Xf) / float(N) - (float(P) / float(N)) * S
        Snew = np.where(h > 0.0, 1.0, np.where(h < 0.0, -1.0, S))
        changed = np.any(Snew != S, axis=1)
        newly_done = active & (~changed)
        iterations[newly_done] = step
        S = Snew
        active &= changed
        if not np.any(active):
            break

    iterations[iterations == 0] = max_steps
    return S.astype(np.int8), iterations


# ---------------------------------------------------------------------
# Analytical benchmark
# ---------------------------------------------------------------------

def expected_m2_m4_iid_rademacher(N: int, mean_y: float) -> Tuple[float, float]:
    """
    y_i in {-1,+1}, iid with E[y_i]=mean_y.
    m = N^{-1} sum_i y_i.
    Returns exact E[m^2] and E[m^4].
    """
    a = float(mean_y)
    EN2 = N + N * (N - 1) * a * a

    # Exact fourth moment of S=sum_i y_i. The a^2 contribution contains
    # both the (3,1) and (2,1,1) index partitions.
    EN4 = (
        N
        + 3 * N * (N - 1)
        + 2 * N * (N - 1) * (3 * N - 4) * a * a
        + N * (N - 1) * (N - 2) * (N - 3) * (a ** 4)
    )
    return EN2 / (N ** 2), EN4 / (N ** 4)


def expected_q2(N: int, mean_pair_spin_product: float) -> float:
    Em2, Em4 = expected_m2_m4_iid_rademacher(N, mean_pair_spin_product)
    return float((N * N * Em4 - 2.0 * N * Em2 + 1.0) / ((N - 1) ** 2))


def analytical_prediction(
    N: int,
    P: int,
    corr: float,
    generator: str,
    lam: float,
    mixture_templates: int,
) -> Dict[str, float]:
    """
    Approximate ensemble prediction.

    For latent, block, and biased-template-like ensembles, pair spin product
    mean is approximately corr^2.

    For a K-template mixture, the pair is in the same cluster with probability
    about 1/K. We mix the same-cluster and independent-cluster q^2 moments.
    """
    q2_null = expected_q2(N, 0.0)
    q2_same = expected_q2(N, corr * corr)

    if generator == "mixture":
        p_same = 1.0 / float(max(1, mixture_templates))
        q2_corr = p_same * q2_same + (1.0 - p_same) * q2_null
    else:
        q2_corr = q2_same

    pred_uncorr_int = math.sqrt(1.0 + lam * q2_null)
    pred_corr_int = math.sqrt(1.0 + lam * q2_corr)
    pred_uncorr_cum = math.sqrt(1.0 + lam * max(P - 1, 0) * q2_null)
    pred_corr_cum = math.sqrt(1.0 + lam * max(P - 1, 0) * q2_corr)

    return {
        "pred_q2_uncorr": q2_null,
        "pred_q2_corr": q2_corr,
        "pred_cost_uncorr_int": pred_uncorr_int,
        "pred_cost_corr_int": pred_corr_int,
        "pred_deltaF_int": pred_corr_int - pred_uncorr_int,
        "pred_cost_uncorr_cum": pred_uncorr_cum,
        "pred_cost_corr_cum": pred_corr_cum,
        "pred_deltaF_cum": pred_corr_cum - pred_uncorr_cum,
    }


# ---------------------------------------------------------------------
# Simulation and checkpointing
# ---------------------------------------------------------------------

def task_key(
    N: int,
    load: float,
    generator: str,
    corr: float,
    rep: int,
) -> str:
    return f"N={N}|load={load:.12g}|gen={generator}|corr={corr:.12g}|rep={rep}"


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_completed(path: Path, key: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(key + "\n")


def simulate_one_condition(
    rng: np.random.Generator,
    X: np.ndarray,
    N: int,
    nominal_load: float,
    generator: str,
    corr: float,
    condition: str,
    rep: int,
    cue_noises: Sequence[float],
    cues_per_noise: int,
    max_steps: int,
    lam: float,
) -> Tuple[List[TrialRow], RealizationRow]:
    P = X.shape[0]
    actual_load = P / float(N)

    cost_int, cost_cum, pair_abs, mean_q2 = geometric_costs(X, lam)

    trials: List[TrialRow] = []
    for noise in cue_noises:
        cues, targets = make_cues(rng, X, float(noise), cues_per_noise)
        recalled, iterations = hopfield_recall_batch(X, cues, max_steps=max_steps)

        initial_overlap = np.mean(cues * X[targets], axis=1)
        final_overlap = np.mean(recalled * X[targets], axis=1)

        for cue_id in range(cues_per_noise):
            target = int(targets[cue_id])
            fin = float(final_overlap[cue_id])
            trials.append(TrialRow(
                N=N,
                nominal_load=float(nominal_load),
                actual_load=float(actual_load),
                P=P,
                generator=generator,
                corr=float(corr),
                condition=condition,
                rep=rep,
                cue_noise=float(noise),
                cue_id=cue_id,
                target_index=target,
                cost_int=float(cost_int[target]),
                cost_cum=float(cost_cum[target]),
                initial_overlap=float(initial_overlap[cue_id]),
                final_overlap=fin,
                retrieval_error=float(1.0 - fin),
                success=int(fin >= 0.90),
                iterations=int(iterations[cue_id]),
            ))

    realization = RealizationRow(
        N=N,
        nominal_load=float(nominal_load),
        actual_load=float(actual_load),
        P=P,
        generator=generator,
        corr=float(corr),
        condition=condition,
        rep=rep,
        mean_pair_overlap_abs=float(pair_abs),
        mean_q2=float(mean_q2),
        mean_cost_int=float(np.mean(cost_int)),
        mean_cost_cum=float(np.mean(cost_cum)),
    )
    return trials, realization


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_trials(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, str]]] = {}
    keys = [
        "N", "nominal_load", "actual_load", "P", "generator",
        "corr", "condition", "cue_noise",
    ]
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for key, rs in sorted(groups.items()):
        record: Dict[str, object] = dict(zip(keys, key))
        record["N"] = int(float(record["N"]))
        record["P"] = int(float(record["P"]))
        for k in ["nominal_load", "actual_load", "corr", "cue_noise"]:
            record[k] = float(record[k])

        for name in [
            "cost_int", "cost_cum", "initial_overlap",
            "final_overlap", "retrieval_error", "success", "iterations",
        ]:
            vals = [float(r[name]) for r in rs]
            m, sem, n = mean_sem(vals)
            record[f"{name}_mean"] = m
            record[f"{name}_sem"] = sem
            record["n_trials"] = n

        costs = [float(r["cost_int"]) for r in rs]
        errors = [float(r["retrieval_error"]) for r in rs]
        successes = [float(r["success"]) for r in rs]
        record["pearson_cost_error"] = pearsonr(costs, errors)
        record["spearman_cost_error"] = spearmanr(costs, errors)
        record["pearson_cost_failure"] = pearsonr(costs, [1.0 - x for x in successes])
        record["spearman_cost_failure"] = spearmanr(costs, [1.0 - x for x in successes])
        out.append(record)
    return out


def summarize_realizations(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, str]]] = {}
    keys = ["N", "nominal_load", "actual_load", "P", "generator", "corr", "condition"]
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for key, rs in sorted(groups.items()):
        record: Dict[str, object] = dict(zip(keys, key))
        record["N"] = int(float(record["N"]))
        record["P"] = int(float(record["P"]))
        for k in ["nominal_load", "actual_load", "corr"]:
            record[k] = float(record[k])

        for name in ["mean_pair_overlap_abs", "mean_q2", "mean_cost_int", "mean_cost_cum"]:
            vals = [float(r[name]) for r in rs]
            m, sem, n = mean_sem(vals)
            record[f"{name}_mean"] = m
            record[f"{name}_sem"] = sem
            record["n_realizations"] = n
        out.append(record)
    return out


def make_delta_summary(
    trial_summary: List[Dict[str, object]],
    realization_summary: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    real_map: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for r in realization_summary:
        key = (
            r["N"], r["nominal_load"], r["actual_load"], r["P"],
            r["generator"], r["corr"],
        )
        real_map.setdefault(key, {})[str(r["condition"])] = r

    trial_map: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for r in trial_summary:
        key = (
            r["N"], r["nominal_load"], r["actual_load"], r["P"],
            r["generator"], r["corr"], r["cue_noise"],
        )
        trial_map.setdefault(key, {})[str(r["condition"])] = r

    out: List[Dict[str, object]] = []
    for key, pair in sorted(trial_map.items()):
        if "correlated" not in pair or "uncorrelated" not in pair:
            continue
        c = pair["correlated"]
        u = pair["uncorrelated"]
        real_key = key[:-1]
        rp = real_map.get(real_key, {})
        rc = rp.get("correlated")
        ru = rp.get("uncorrelated")
        if rc is None or ru is None:
            continue

        delta_cost_int = float(rc["mean_cost_int_mean"]) - float(ru["mean_cost_int_mean"])
        delta_cost_cum = float(rc["mean_cost_cum_mean"]) - float(ru["mean_cost_cum_mean"])
        retrieval_degradation = float(u["final_overlap_mean"]) - float(c["final_overlap_mean"])
        failure_increase = (1.0 - float(c["success_mean"])) - (1.0 - float(u["success_mean"]))

        out.append({
            "N": key[0],
            "nominal_load": key[1],
            "actual_load": key[2],
            "P": key[3],
            "generator": key[4],
            "corr": key[5],
            "cue_noise": key[6],
            "deltaF_int": delta_cost_int,
            "deltaF_cum": delta_cost_cum,
            "retrieval_degradation": retrieval_degradation,
            "failure_probability_increase": failure_increase,
            "corr_final_overlap_mean": c["final_overlap_mean"],
            "uncorr_final_overlap_mean": u["final_overlap_mean"],
            "corr_success_mean": c["success_mean"],
            "uncorr_success_mean": u["success_mean"],
            "corr_spearman_cost_error": c["spearman_cost_error"],
            "uncorr_spearman_cost_error": u["spearman_cost_error"],
        })
    return out


def analytical_rows(
    realization_summary: List[Dict[str, object]],
    lam: float,
    mixture_templates: int,
) -> List[Dict[str, object]]:
    keys = sorted({
        (
            int(r["N"]), float(r["nominal_load"]), float(r["actual_load"]),
            int(r["P"]), str(r["generator"]), float(r["corr"])
        )
        for r in realization_summary
    })

    # observed correlated and uncorrelated means
    obs: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for r in realization_summary:
        key = (
            int(r["N"]), float(r["nominal_load"]), float(r["actual_load"]),
            int(r["P"]), str(r["generator"]), float(r["corr"])
        )
        obs.setdefault(key, {})[str(r["condition"])] = r

    out: List[Dict[str, object]] = []
    for N, nload, aload, P, generator, corr in keys:
        pair = obs.get((N, nload, aload, P, generator, corr), {})
        if "correlated" not in pair or "uncorrelated" not in pair:
            continue
        pred = analytical_prediction(N, P, corr, generator, lam, mixture_templates)
        observed_delta_int = (
            float(pair["correlated"]["mean_cost_int_mean"])
            - float(pair["uncorrelated"]["mean_cost_int_mean"])
        )
        observed_delta_cum = (
            float(pair["correlated"]["mean_cost_cum_mean"])
            - float(pair["uncorrelated"]["mean_cost_cum_mean"])
        )
        out.append({
            "N": N,
            "nominal_load": nload,
            "actual_load": aload,
            "P": P,
            "generator": generator,
            "corr": corr,
            "observed_deltaF_int": observed_delta_int,
            "predicted_deltaF_int": pred["pred_deltaF_int"],
            "observed_deltaF_cum": observed_delta_cum,
            "predicted_deltaF_cum": pred["pred_deltaF_cum"],
            "pred_q2_uncorr": pred["pred_q2_uncorr"],
            "pred_q2_corr": pred["pred_q2_corr"],
        })
    return out


def pooled_delta_correlations(delta_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []

    grouping_specs = [
        ("all", []),
        ("by_generator", ["generator"]),
        ("by_cue_noise", ["cue_noise"]),
        ("by_generator_and_noise", ["generator", "cue_noise"]),
    ]

    for label, fields in grouping_specs:
        groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
        for r in delta_rows:
            key = tuple(r[f] for f in fields)
            groups.setdefault(key, []).append(r)

        for key, rs in sorted(groups.items()):
            x = [float(r["deltaF_int"]) for r in rs]
            y1 = [float(r["retrieval_degradation"]) for r in rs]
            y2 = [float(r["failure_probability_increase"]) for r in rs]
            record: Dict[str, object] = {
                "grouping": label,
                "n_conditions": len(rs),
                "pearson_deltaF_vs_retrieval_degradation": pearsonr(x, y1),
                "spearman_deltaF_vs_retrieval_degradation": spearmanr(x, y1),
                "pearson_deltaF_vs_failure_increase": pearsonr(x, y2),
                "spearman_deltaF_vs_failure_increase": spearmanr(x, y2),
            }
            for field, value in zip(fields, key):
                record[field] = value
            out.append(record)
    return out


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def plot_delta_cost_vs_retrieval(delta_rows: List[Dict[str, object]], figs_dir: Path) -> None:
    if not delta_rows:
        return
    x = np.array([float(r["deltaF_int"]) for r in delta_rows])
    y = np.array([float(r["retrieval_degradation"]) for r in delta_rows])

    plt.figure(figsize=(7.0, 4.8))
    for generator in sorted({str(r["generator"]) for r in delta_rows}):
        rows = [r for r in delta_rows if str(r["generator"]) == generator]
        plt.scatter(
            [float(r["deltaF_int"]) for r in rows],
            [float(r["retrieval_degradation"]) for r in rows],
            s=20,
            alpha=0.65,
            label=generator,
        )

    if x.size >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        plt.plot(xx, intercept + slope * xx, linewidth=1.5, label="pooled linear fit")

    rho = spearmanr(x, y)
    plt.xlabel(r"Excess intensive writing cost $\Delta F_{\mathrm{int}}$")
    plt.ylabel(r"Retrieval degradation $\Delta(1-m_{\mathrm{ret}})$")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, framealpha=0.55)
    plt.tight_layout()
    plt.savefig(figs_dir / "fig_R1_deltaF_vs_retrieval_degradation.png", dpi=260)
    plt.close()

    (figs_dir / "fig_R1_spearman_value.txt").write_text(
        f"Spearman correlation between deltaF_int and retrieval degradation: {rho:.8g}\n",
        encoding="utf-8",
    )


def plot_retrieval_vs_noise(
    trial_summary: List[Dict[str, object]],
    figs_dir: Path,
    target_N: int,
    target_load: float,
    target_generator: str,
) -> None:
    rows = [
        r for r in trial_summary
        if int(r["N"]) == target_N
        and str(r["generator"]) == target_generator
        and abs(float(r["actual_load"]) - target_load) < 0.03
    ]
    if not rows:
        return

    corrs = sorted({float(r["corr"]) for r in rows})
    plt.figure(figsize=(7.0, 4.8))
    for corr in corrs:
        rr = sorted(
            [
                r for r in rows
                if float(r["corr"]) == corr and str(r["condition"]) == "correlated"
            ],
            key=lambda r: float(r["cue_noise"]),
        )
        if not rr:
            continue
        plt.errorbar(
            [float(r["cue_noise"]) for r in rr],
            [float(r["final_overlap_mean"]) for r in rr],
            yerr=[float(r["final_overlap_sem"]) for r in rr],
            marker="o",
            linewidth=1.3,
            capsize=3,
            label=rf"$c={corr:g}$",
        )

    plt.xlabel("Cue corruption probability")
    plt.ylabel(r"Final retrieval overlap $m_{\mathrm{ret}}$")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, framealpha=0.55)
    plt.tight_layout()
    plt.savefig(figs_dir / f"fig_R2_retrieval_vs_noise_{target_generator}_N{target_N}.png", dpi=260)
    plt.close()


def plot_analytic_vs_simulation(analytic: List[Dict[str, object]], figs_dir: Path) -> None:
    if not analytic:
        return
    x = np.array([float(r["predicted_deltaF_int"]) for r in analytic])
    y = np.array([float(r["observed_deltaF_int"]) for r in analytic])

    plt.figure(figsize=(6.2, 5.2))
    for generator in sorted({str(r["generator"]) for r in analytic}):
        rows = [r for r in analytic if str(r["generator"]) == generator]
        plt.scatter(
            [float(r["predicted_deltaF_int"]) for r in rows],
            [float(r["observed_deltaF_int"]) for r in rows],
            s=24,
            alpha=0.7,
            label=generator,
        )

    low = min(float(np.min(x)), float(np.min(y)))
    high = max(float(np.max(x)), float(np.max(y)))
    plt.plot([low, high], [low, high], linewidth=1.2, label="identity")
    plt.xlabel(r"Analytical prediction for $\Delta F_{\mathrm{int}}$")
    plt.ylabel(r"Simulated $\Delta F_{\mathrm{int}}$")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, framealpha=0.55)
    plt.tight_layout()
    plt.savefig(figs_dir / "fig_R3_analytic_vs_simulated_deltaF.png", dpi=260)
    plt.close()


def plot_finite_size_retrieval_link(
    delta_rows: List[Dict[str, object]],
    figs_dir: Path,
    target_generator: str,
    target_corr: float,
    target_load: float,
    target_noise: float,
) -> None:
    rows = [
        r for r in delta_rows
        if str(r["generator"]) == target_generator
        and abs(float(r["corr"]) - target_corr) < 1e-12
        and abs(float(r["actual_load"]) - target_load) < 0.03
        and abs(float(r["cue_noise"]) - target_noise) < 1e-12
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(r["N"]))

    Ns = np.array([int(r["N"]) for r in rows], dtype=float)
    dc = np.array([float(r["deltaF_int"]) for r in rows])
    dr = np.array([float(r["retrieval_degradation"]) for r in rows])

    fig, ax1 = plt.subplots(figsize=(7.0, 4.8))
    ax1.plot(Ns, dc, marker="o", linewidth=1.4, label=r"$\Delta F_{\mathrm{int}}$")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"System size $N$")
    ax1.set_ylabel(r"$\Delta F_{\mathrm{int}}$")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(Ns, dr, marker="s", linewidth=1.4, linestyle="--", label="retrieval degradation")
    ax2.set_ylabel("Retrieval degradation")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, framealpha=0.55)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_R4_finite_size_cost_and_retrieval.png", dpi=260)
    plt.close(fig)


def plot_success_heatmap(
    trial_summary: List[Dict[str, object]],
    figs_dir: Path,
    target_N: int,
    target_load: float,
    target_generator: str,
) -> None:
    rows = [
        r for r in trial_summary
        if int(r["N"]) == target_N
        and str(r["generator"]) == target_generator
        and str(r["condition"]) == "correlated"
        and abs(float(r["actual_load"]) - target_load) < 0.03
    ]
    if not rows:
        return

    corrs = sorted({float(r["corr"]) for r in rows})
    noises = sorted({float(r["cue_noise"]) for r in rows})
    matrix = np.full((len(corrs), len(noises)), np.nan)
    for r in rows:
        a = corrs.index(float(r["corr"]))
        b = noises.index(float(r["cue_noise"]))
        matrix[a, b] = float(r["success_mean"])

    plt.figure(figsize=(7.0, 4.8))
    image = plt.imshow(matrix, aspect="auto", origin="lower")
    plt.colorbar(image, label="Retrieval success probability")
    plt.xticks(np.arange(len(noises)), [f"{x:g}" for x in noises])
    plt.yticks(np.arange(len(corrs)), [f"{x:g}" for x in corrs])
    plt.xlabel("Cue corruption probability")
    plt.ylabel("Correlation parameter")
    plt.tight_layout()
    plt.savefig(figs_dir / f"fig_R5_success_heatmap_{target_generator}_N{target_N}.png", dpi=260)
    plt.close()


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def write_readme(outdir: Path, args: argparse.Namespace) -> None:
    text = f"""L4 special-issue retrieval/geometry simulation
================================================

Purpose
-------
Relate the finite-size geometric Hebbian writing cost to actual Hopfield
retrieval degradation and compare the simulated cost with an overlap-moment
analytical prediction.

Parameters
----------
Ns: {args.Ns}
loads: {args.loads}
generators: {args.generators}
corrs: {args.corrs}
cue_noises: {args.cue_noises}
reps: {args.reps}
cues_per_noise: {args.cues_per_noise}
max_steps: {args.max_steps}
lambda_interference: {args.lambda_interference}
success_threshold: fixed at final overlap >= 0.90
seed: {args.seed}

Important output files
----------------------
data/trial_level_retrieval.csv
data/realization_level_geometry.csv
data/trial_condition_summary.csv
data/realization_condition_summary.csv
data/correlated_minus_uncorrelated_summary.csv
data/analytical_prediction_comparison.csv
data/pooled_cost_retrieval_correlations.csv

figs/fig_R1_deltaF_vs_retrieval_degradation.png
figs/fig_R2_retrieval_vs_noise_*.png
figs/fig_R3_analytic_vs_simulated_deltaF.png
figs/fig_R4_finite_size_cost_and_retrieval.png
figs/fig_R5_success_heatmap_*.png

Checkpoint and resume
---------------------
Each completed realization is recorded in:

data/completed_tasks.txt

Rerunning the same command resumes from unfinished tasks. Use --restart to
delete prior CSV/checkpoint outputs and begin again.

Interpretation
--------------
A positive association between deltaF_int and retrieval degradation supports
the use of the geometric cost as an AI-relevant diagnostic.

A weak or absent association is still informative: the cost then describes
writing interference but should not be presented as a direct predictor of
retrieval performance.

The analytical benchmark is an overlap-moment approximation. It is expected
to work best for latent and block ensembles, and more approximately for the
mixture ensemble.
"""
    (outdir / "README.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L4: Hebbian writing geometry versus Hopfield retrieval."
    )
    p.add_argument("--outdir", default="L4_special_issue_retrieval_geometry_results")
    p.add_argument("--Ns", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--loads", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30])
    p.add_argument("--generators", nargs="+", default=["latent", "block", "mixture"])
    p.add_argument("--corrs", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45])
    p.add_argument("--cue-noises", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--cues-per-noise", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--lambda-interference", type=float, default=2.0)
    p.add_argument("--mixture-templates", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--restart", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.outdir = "L4_special_issue_retrieval_geometry_quick"
        args.Ns = [64, 128]
        args.loads = [0.10, 0.20]
        args.generators = ["latent", "mixture"]
        args.corrs = [0.25, 0.35]
        args.cue_noises = [0.10, 0.20]
        args.reps = 2
        args.cues_per_noise = 4
        args.max_steps = 8

    for g in args.generators:
        if g not in GENERATORS:
            raise ValueError(f"Unknown generator '{g}'. Available: {sorted(GENERATORS)}")

    outdir = Path(args.outdir)
    data_dir, figs_dir = ensure_dirs(outdir)

    trial_path = data_dir / "trial_level_retrieval.csv"
    realization_path = data_dir / "realization_level_geometry.csv"
    completed_path = data_dir / "completed_tasks.txt"

    if args.restart:
        for path in [trial_path, realization_path, completed_path]:
            if path.exists():
                path.unlink()

    completed = load_completed(completed_path)
    rng = np.random.default_rng(args.seed)

    total = (
        len(args.Ns) * len(args.loads) * len(args.generators)
        * len(args.corrs) * args.reps
    )
    done_now = 0
    skipped = 0

    trial_fields = list(TrialRow.__dataclass_fields__.keys())
    realization_fields = list(RealizationRow.__dataclass_fields__.keys())

    for N in args.Ns:
        for load in args.loads:
            P = max(1, int(round(N * load)))
            for generator in args.generators:
                for corr in args.corrs:
                    for rep in range(args.reps):
                        key = task_key(N, load, generator, corr, rep)
                        if key in completed:
                            skipped += 1
                            continue

                        # A process-independent deterministic child seed keeps
                        # resumed runs identical across Python sessions.
                        seed_material = (
                            f"{args.seed}|{N}|{load:.17g}|{generator}|"
                            f"{corr:.17g}|{rep}"
                        ).encode("utf-8")
                        digest = hashlib.sha256(seed_material).digest()
                        child_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
                        local_rng = np.random.default_rng(child_seed)

                        X_corr = GENERATORS[generator](local_rng, P, N, corr)
                        X_null = patterns_uncorrelated(local_rng, P, N)

                        trials_c, real_c = simulate_one_condition(
                            local_rng, X_corr, N, load, generator, corr,
                            "correlated", rep, args.cue_noises,
                            args.cues_per_noise, args.max_steps,
                            args.lambda_interference,
                        )
                        trials_u, real_u = simulate_one_condition(
                            local_rng, X_null, N, load, generator, corr,
                            "uncorrelated", rep, args.cue_noises,
                            args.cues_per_noise, args.max_steps,
                            args.lambda_interference,
                        )

                        append_dataclass_rows(
                            trial_path, trials_c + trials_u, trial_fields
                        )
                        append_dataclass_rows(
                            realization_path, [real_c, real_u], realization_fields
                        )
                        mark_completed(completed_path, key)
                        completed.add(key)
                        done_now += 1

            print(
                f"Completed N={N}, load={load:g}; "
                f"new tasks={done_now}, skipped={skipped}, total={total}",
                flush=True,
            )

    trial_rows = read_csv(trial_path)
    realization_rows = read_csv(realization_path)

    trial_summary = summarize_trials(trial_rows)
    realization_summary = summarize_realizations(realization_rows)
    delta_summary = make_delta_summary(trial_summary, realization_summary)
    analytic = analytical_rows(
        realization_summary,
        args.lambda_interference,
        args.mixture_templates,
    )
    pooled = pooled_delta_correlations(delta_summary)

    write_dict_rows(data_dir / "trial_condition_summary.csv", trial_summary)
    write_dict_rows(data_dir / "realization_condition_summary.csv", realization_summary)
    write_dict_rows(data_dir / "correlated_minus_uncorrelated_summary.csv", delta_summary)
    write_dict_rows(data_dir / "analytical_prediction_comparison.csv", analytic)
    write_dict_rows(data_dir / "pooled_cost_retrieval_correlations.csv", pooled)

    if trial_summary:
        Ns_available = sorted({int(r["N"]) for r in trial_summary})
        loads_available = sorted({float(r["actual_load"]) for r in trial_summary})
        corrs_available = sorted({float(r["corr"]) for r in trial_summary})
        noises_available = sorted({float(r["cue_noise"]) for r in trial_summary})

        target_N = max(Ns_available)
        target_load = nearest(0.20, loads_available)
        target_corr = nearest(0.35, corrs_available)
        target_noise = nearest(0.15, noises_available)
        target_generator = "latent" if "latent" in args.generators else args.generators[0]

        plot_delta_cost_vs_retrieval(delta_summary, figs_dir)
        plot_retrieval_vs_noise(
            trial_summary, figs_dir, target_N, target_load, target_generator
        )
        plot_analytic_vs_simulation(analytic, figs_dir)
        plot_finite_size_retrieval_link(
            delta_summary, figs_dir, target_generator,
            target_corr, target_load, target_noise,
        )
        plot_success_heatmap(
            trial_summary, figs_dir, target_N, target_load, target_generator
        )

    write_readme(outdir, args)

    print("Simulation and summary generation complete.")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Trial rows: {len(trial_rows)}")
    print(f"Realization rows: {len(realization_rows)}")
    print(f"Newly completed tasks: {done_now}")
    print(f"Previously completed tasks skipped: {skipped}")


if __name__ == "__main__":
    main()
