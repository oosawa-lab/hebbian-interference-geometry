#!/usr/bin/env python3
"""Generate the latent-template asymptotic benchmark from robustness CSV data.

For patterns xi_i^mu=t_i epsilon_i^mu with E[epsilon]=c, the pair overlap
converges to c^2. For normalized upper-triangular Hebbian directions,
the direction overlap converges to c^4; therefore the squared interference
term converges to c^8 and

    Delta F_int -> sqrt(1 + lambda*c^8) - 1

relative to the uncorrelated baseline.
"""
from pathlib import Path
import argparse, math
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='../robustness/data/L4_V04_robustness_nearest_load_scaling.csv')
    ap.add_argument('--out',default='../figs/fig4_latent_asymptotic_benchmark')
    ap.add_argument('--lambda-interference',type=float,default=2.0)
    ap.add_argument('--reference-load',type=float,default=0.60)
    ap.add_argument('--corrs',type=float,nargs='+',default=[0.25,0.35,0.45])
    args=ap.parse_args()
    df=pd.read_csv(args.csv)
    df=df[(df.generator=='latent')&(df.lambda_interference==args.lambda_interference)&(df.randers_mode=='mean')&(df.reference_load==args.reference_load)]
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for c,marker in zip(args.corrs,['o','s','^','d']):
        s=df[df['corr']==c].sort_values('N')
        x=s.N.to_numpy(float); y=s.deltaF_int_mean.to_numpy(float)
        lo=s.deltaF_int_ci95_lo.to_numpy(float); hi=s.deltaF_int_ci95_hi.to_numpy(float)
        ax.errorbar(x,y,yerr=[y-lo,hi-y],marker=marker,linewidth=1.5,capsize=3,label=rf'$c={c:.2f}$')
        asym=math.sqrt(1+args.lambda_interference*c**8)-1
        ax.axhline(asym,ls='--',lw=1.1,label=rf'$\\Delta F_\\infty(c={c:.2f})$')
    ax.set_xscale('log',base=2); ax.set_xlabel(r'System size $N$'); ax.set_ylabel(r'Intensive excess $\\Delta F_{\\rm int}$')
    ax.grid(True,alpha=.3,which='both'); ax.legend(frameon=True,framealpha=.72,fontsize=8,ncol=2); fig.tight_layout()
    fig.savefig(args.out+'.png',dpi=260); fig.savefig(args.out+'.pdf')
if __name__=='__main__': main()
