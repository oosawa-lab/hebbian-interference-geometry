#!/usr/bin/env python3
"""
L4/V04 local simulation: Hopfield Hebbian directions with Hessian/Finsler costs.

Generates CSV data and figures for:
  Fig.2  intensive writing cost with error bars
  Fig.3  intensive vs cumulative excess cost
  Fig.4  finite-size scaling with nearest-load matching
  Fig.5  Randers writing/erasing asymmetry
  Fig.S1 intensive susceptibility

Dependencies: numpy, matplotlib.
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def mkdirs(outdir: Path):
    data = outdir / 'data'; figs = outdir / 'figs'
    data.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    return data, figs


def patterns_uncorr(rng, P, N):
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(P, N))


def patterns_corr(rng, P, N, corr):
    corr = float(np.clip(corr, -0.999, 0.999))
    template = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    flip_prob = (1.0 - corr) / 2.0
    flips = rng.random((P, N)) < flip_prob
    X = np.tile(template, (P, 1)).astype(np.int8)
    X[flips] *= -1
    return X


def hebb_dir(x):
    N = x.size
    iu = np.triu_indices(N, k=1)
    v = np.outer(x, x).astype(float)[iu]
    v /= math.sqrt(v.size)
    return v


def dirs(patterns):
    return np.vstack([hebb_dir(patterns[i]) for i in range(patterns.shape[0])])


def costs(stored, query, lam):
    base = float(query @ query)
    if stored.shape[0] == 0:
        return math.sqrt(base), math.sqrt(base)
    ov = stored @ query
    sq = ov * ov
    cum = base + lam * float(np.sum(sq))
    inte = base + lam * float(np.mean(sq))
    return math.sqrt(max(cum, 0.0)), math.sqrt(max(inte, 0.0))


def randers_A(stored, query, alpha):
    if stored.shape[0] == 0 or alpha == 0:
        return 0.0
    m = np.mean(stored, axis=0)
    nrm = float(np.linalg.norm(m))
    if nrm < 1e-14:
        return 0.0
    return float(alpha * ((m / nrm) @ query))


def mean_sem(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan'), 0
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0, int(a.size)


def write_csv(rows, path):
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def simulate(args):
    rng = np.random.default_rng(args.seed)
    trial_rows = []
    for N in args.Ns:
        for nominal_load in args.loads:
            P = max(1, int(round(N * nominal_load)))
            actual_load = P / N
            for condition in ['uncorrelated', 'correlated']:
                for rep in range(args.reps):
                    pats = patterns_uncorr(rng, P + 1, N) if condition == 'uncorrelated' else patterns_corr(rng, P + 1, N, args.corr)
                    D = dirs(pats)
                    stored, query = D[:P], D[P]
                    Fcum, Fint = costs(stored, query, args.lambda_interference)
                    A = randers_A(stored, query, args.randers_alpha)
                    trial_rows.append(dict(
                        N=N, nominal_load=nominal_load, actual_load=actual_load, P=P,
                        condition=condition, rep=rep,
                        cost_cum=Fcum, cost_int=Fint,
                        randers_write=Fint + A, randers_erase=Fint - A,
                        randers_asymmetry=2.0 * A,
                    ))
    return trial_rows


def summarize(trial_rows):
    groups = defaultdict(list)
    for r in trial_rows:
        groups[(r['N'], r['nominal_load'], r['actual_load'], r['P'], r['condition'])].append(r)
    summary = []
    for (N, nominal_load, actual_load, P, condition), rs in sorted(groups.items()):
        row = dict(N=N, nominal_load=nominal_load, actual_load=actual_load, P=P, condition=condition)
        for key in ['cost_cum', 'cost_int', 'randers_write', 'randers_erase', 'randers_asymmetry']:
            m, s, n = mean_sem([x[key] for x in rs])
            row[key + '_mean'] = m; row[key + '_sem'] = s; row['n_reps'] = n
        summary.append(row)
    return summary


def delta_summary(summary):
    by = defaultdict(dict)
    for r in summary:
        by[(r['N'], r['nominal_load'], r['actual_load'], r['P'])][r['condition']] = r
    out = []
    for (N, nominal_load, actual_load, P), d in sorted(by.items()):
        if 'correlated' not in d or 'uncorrelated' not in d: continue
        c, u = d['correlated'], d['uncorrelated']
        out.append(dict(
            N=N, nominal_load=nominal_load, actual_load=actual_load, P=P,
            deltaF_cum_mean=c['cost_cum_mean'] - u['cost_cum_mean'],
            deltaF_cum_sem=math.sqrt(c['cost_cum_sem']**2 + u['cost_cum_sem']**2),
            deltaF_int_mean=c['cost_int_mean'] - u['cost_int_mean'],
            deltaF_int_sem=math.sqrt(c['cost_int_sem']**2 + u['cost_int_sem']**2),
            delta_randers_asymmetry_mean=c['randers_asymmetry_mean'] - u['randers_asymmetry_mean'],
            delta_randers_asymmetry_sem=math.sqrt(c['randers_asymmetry_sem']**2 + u['randers_asymmetry_sem']**2),
            corr_cost_int_mean=c['cost_int_mean'], uncorr_cost_int_mean=u['cost_int_mean'],
            corr_cost_cum_mean=c['cost_cum_mean'], uncorr_cost_cum_mean=u['cost_cum_mean'],
            corr_randers_asymmetry_mean=c['randers_asymmetry_mean'],
            uncorr_randers_asymmetry_mean=u['randers_asymmetry_mean'],
        ))
    return out


def nearest_scaling(delta_rows, ref_loads, tol):
    byN = defaultdict(list)
    for r in delta_rows: byN[r['N']].append(r)
    out = []
    for ref in ref_loads:
        for N, rows in sorted(byN.items()):
            best = min(rows, key=lambda r: abs(r['actual_load'] - ref))
            dist = abs(best['actual_load'] - ref)
            if tol >= 0 and dist > tol: continue
            row = dict(best); row['reference_load'] = ref; row['load_distance'] = dist
            out.append(row)
    return out


def susceptibility(delta_rows):
    byN = defaultdict(list)
    for r in delta_rows: byN[r['N']].append(r)
    out = []
    for N, rows in sorted(byN.items()):
        rows = sorted(rows, key=lambda r: r['actual_load'])
        if len(rows) < 3: continue
        x = np.array([r['actual_load'] for r in rows], float)
        y = np.array([r['deltaF_int_mean'] for r in rows], float)
        chi = np.gradient(y, x)
        for r, ch in zip(rows, chi):
            out.append(dict(N=N, nominal_load=r['nominal_load'], actual_load=r['actual_load'], chiF_int=float(ch), deltaF_int_mean=r['deltaF_int_mean']))
    return out


def plot_fig2(summary, figs):
    maxN = max(r['N'] for r in summary)
    sub = [r for r in summary if r['N'] == maxN]
    plt.figure(figsize=(7,4.6))
    for cond in ['uncorrelated', 'correlated']:
        rows = sorted([r for r in sub if r['condition'] == cond], key=lambda r: r['actual_load'])
        plt.errorbar([r['actual_load'] for r in rows], [r['cost_int_mean'] for r in rows],
                     yerr=[r['cost_int_sem'] for r in rows], marker='o', linewidth=1.5, capsize=3, label=cond)
    plt.xlabel(r'Memory load $P/N$'); plt.ylabel(r'Intensive writing cost $F_{\mathrm{int}}$')
    plt.grid(True, alpha=.3); plt.legend(frameon=True, framealpha=.75); plt.tight_layout()
    plt.savefig(figs/'fig2_V04_intensive_cost_errorbars.png', dpi=240); plt.savefig(figs/'fig2_V04_intensive_cost_errorbars.pdf'); plt.close()


def plot_fig3(delta, figs):
    maxN = max(r['N'] for r in delta)
    rows = sorted([r for r in delta if r['N'] == maxN], key=lambda r: r['actual_load'])
    x = [r['actual_load'] for r in rows]
    plt.figure(figsize=(7,4.6))
    plt.errorbar(x, [r['deltaF_int_mean'] for r in rows], yerr=[r['deltaF_int_sem'] for r in rows], marker='o', linewidth=1.5, capsize=3, label=r'$\Delta F_{\mathrm{int}}$')
    plt.errorbar(x, [r['deltaF_cum_mean'] for r in rows], yerr=[r['deltaF_cum_sem'] for r in rows], marker='s', linewidth=1.5, capsize=3, label=r'$\Delta F_{\mathrm{cum}}$')
    plt.xlabel(r'Memory load $P/N$'); plt.ylabel('Correlation-induced excess cost')
    plt.grid(True, alpha=.3); plt.legend(frameon=True, framealpha=.75); plt.tight_layout()
    plt.savefig(figs/'fig3_V04_deltaF_intensive_vs_cumulative.png', dpi=240); plt.savefig(figs/'fig3_V04_deltaF_intensive_vs_cumulative.pdf'); plt.close()


def plot_fig4(scale, figs):
    plt.figure(figsize=(7,4.6))
    for ref in sorted(set(r['reference_load'] for r in scale)):
        rows = sorted([r for r in scale if r['reference_load'] == ref], key=lambda r: r['N'])
        plt.errorbar([r['N'] for r in rows], [r['deltaF_int_mean'] for r in rows], yerr=[r['deltaF_int_sem'] for r in rows], marker='o', linewidth=1.5, capsize=3, label=rf'$P/N \approx {ref:g}$')
    plt.xscale('log', base=2); plt.xlabel(r'System size $N$'); plt.ylabel(r'Intensive excess cost $\Delta F_{\mathrm{int}}$')
    plt.grid(True, alpha=.3, which='both'); plt.legend(frameon=True, framealpha=.75); plt.tight_layout()
    plt.savefig(figs/'fig4_V04_finite_size_scaling_nearest_load.png', dpi=240); plt.savefig(figs/'fig4_V04_finite_size_scaling_nearest_load.pdf'); plt.close()


def plot_fig5(delta, figs):
    maxN = max(r['N'] for r in delta)
    rows = sorted([r for r in delta if r['N'] == maxN], key=lambda r: r['actual_load'])
    x = [r['actual_load'] for r in rows]
    plt.figure(figsize=(7,4.6))
    plt.plot(x, [r['corr_randers_asymmetry_mean'] for r in rows], marker='o', linewidth=1.5, label='correlated')
    plt.plot(x, [r['uncorr_randers_asymmetry_mean'] for r in rows], marker='s', linewidth=1.5, label='uncorrelated')
    plt.errorbar(x, [r['delta_randers_asymmetry_mean'] for r in rows], yerr=[r['delta_randers_asymmetry_sem'] for r in rows], marker='^', linewidth=1.5, capsize=3, label='excess')
    plt.axhline(0, linewidth=.8); plt.xlabel(r'Memory load $P/N$'); plt.ylabel(r'Randers asymmetry $F_R(V)-F_R(-V)$')
    plt.grid(True, alpha=.3); plt.legend(frameon=True, framealpha=.75); plt.tight_layout()
    plt.savefig(figs/'fig5_V04_randers_asymmetry.png', dpi=240); plt.savefig(figs/'fig5_V04_randers_asymmetry.pdf'); plt.close()


def plot_supp(chi, figs):
    plt.figure(figsize=(7,4.6))
    for N in sorted(set(r['N'] for r in chi)):
        rows = sorted([r for r in chi if r['N'] == N], key=lambda r: r['actual_load'])
        plt.plot([r['actual_load'] for r in rows], [r['chiF_int'] for r in rows], marker='o', linewidth=1.2, label=rf'$N={N}$')
    plt.axhline(0, linewidth=.8); plt.xlabel(r'Memory load $P/N$'); plt.ylabel(r'$\chi_F=d\Delta F_{\mathrm{int}}/d(P/N)$')
    plt.grid(True, alpha=.3); plt.legend(frameon=True, framealpha=.75, fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(figs/'figS1_V04_intensive_susceptibility.png', dpi=240); plt.savefig(figs/'figS1_V04_intensive_susceptibility.pdf'); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='L4_V04_local_results')
    p.add_argument('--Ns', type=int, nargs='+', default=[64,128,256,512,1024])
    p.add_argument('--loads', type=float, nargs='+', default=[.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60])
    p.add_argument('--reps', type=int, default=120)
    p.add_argument('--corr', type=float, default=.35)
    p.add_argument('--lambda-interference', type=float, default=2.0)
    p.add_argument('--randers-alpha', type=float, default=.15)
    p.add_argument('--reference-loads', type=float, nargs='+', default=[.10,.30,.60])
    p.add_argument('--nearest-load-tolerance', type=float, default=.02)
    p.add_argument('--seed', type=int, default=20260710)
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.outdir='L4_V04_local_results_quick'; args.Ns=[32,64,128]; args.loads=[.10,.30,.60]; args.reps=10; args.nearest_load_tolerance=.04
    out = Path(args.outdir); data, figs = mkdirs(out)
    rows = simulate(args); write_csv(rows, data/'L4_V04_trial_level_costs.csv')
    summ = summarize(rows); write_csv(summ, data/'L4_V04_cost_summary.csv')
    delta = delta_summary(summ); write_csv(delta, data/'L4_V04_deltaF_summary.csv')
    scale = nearest_scaling(delta, args.reference_loads, args.nearest_load_tolerance); write_csv(scale, data/'L4_V04_finite_size_nearest_load_summary.csv')
    chi = susceptibility(delta); write_csv(chi, data/'L4_V04_intensive_susceptibility.csv')
    plot_fig2(summ, figs); plot_fig3(delta, figs); plot_fig4(scale, figs); plot_fig5(delta, figs); plot_supp(chi, figs)
    (out/'README_L4_V04_local_output.txt').write_text(
        'L4/V04 local simulation complete. Main figures are in figs/. Main CSV files are in data/.\n'
        'Use Fig.2-Fig.3 for the main intensive/cumulative cost result, Fig.4 for nearest-load finite-size check, and Fig.5 for Randers asymmetry.\n', encoding='utf-8')
    print('Complete:', out.resolve())
    print('Main figures written to:', figs.resolve())

if __name__ == '__main__':
    main()
