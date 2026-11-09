#!/usr/bin/env python3
"""Make publication-readable SHAP rank bump charts using exact Stage 3 data.

Unlike the v1 chart, feature labels are in a separate legend and never
placed on top of overlapping lines. One model per chart.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ['CCA','Median/Mode','KNN k=5']
FEATURES = [
    'RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','INDFMPIR',
    'BMXBMI','BMXWAIST','BP_SYS_MEAN','BP_DIA_MEAN','SMQ020','PAQ605',
]


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--importance',type=Path,
        default=Path('output/stage3_final/shap/shap_global_importance_final.csv'))
    parser.add_argument('--out',type=Path,
        default=Path('output/stage3_reporting_v2_figures'))
    args=parser.parse_args()
    inp=args.importance if args.importance.is_absolute() else root/args.importance
    out=args.out if args.out.is_absolute() else root/args.out
    if not inp.is_file(): raise FileNotFoundError(inp)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}')
    df=pd.read_csv(inp)
    required={'split_id','model','method','feature','rank','mean_abs_shap'}
    if required-set(df.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(df.columns))}')
    if len(df)!=1650: raise ValueError(f'Expected 1650 rows, found {len(df)}')
    if df.duplicated(['split_id','model','method','feature']).any():
        raise ValueError('Duplicate importance rows')
    if set(df.model)!={'LR','RF'} or set(df.method)!=set(METHODS) or set(df.feature)!=set(FEATURES):
        raise ValueError('Unexpected models, methods or features')
    if df[['rank','mean_abs_shap']].isna().any().any():
        raise ValueError('Missing feature rank or importance')
    if not df.groupby(['model','method','feature'])['split_id'].nunique().eq(25).all():
        raise ValueError('Expected 25 splits for every model / method / feature')
    out.mkdir(parents=True,exist_ok=True)
    avg=(df.groupby(['model','method','feature'],as_index=False)
           .agg(mean_rank=('rank','mean'),sd_rank=('rank','std')))
    avg.to_csv(out/'mean_feature_rank_by_method.csv',index=False)
    x=np.arange(len(METHODS))
    palette=plt.get_cmap('tab20').colors
    markers=['o','s','^','D','v','P','X','>','<','h','*']
    for model in ['LR','RF']:
        fig,ax=plt.subplots(figsize=(12.2,6.3))
        for feature_idx,feature in enumerate(FEATURES):
            sel=(avg[(avg.model==model)&(avg.feature==feature)]
                 .set_index('method').reindex(METHODS))
            y=sel.mean_rank.to_numpy(dtype=float)
            if not np.isfinite(y).all(): raise ValueError('Missing mean rank')
            ax.plot(x,y,marker=markers[feature_idx],markersize=6,linewidth=2,
                    color=palette[feature_idx],label=feature)
        ax.set_xticks(x,METHODS,fontsize=11)
        ax.set_yticks(range(1,12))
        ax.set_ylim(11.5,.5)
        ax.set_xlim(-.12,2.12)
        ax.set_ylabel('Mean rank over 25 outer splits (1 = highest importance)',fontsize=11)
        ax.set_title(f'SHAP importance rank by missing-data method: {model}',fontsize=15)
        ax.grid(axis='y',alpha=.26)
        ax.legend(title='Original predictor',loc='center left',
                  bbox_to_anchor=(1.02,.5),fontsize=10,frameon=False)
        fig.subplots_adjust(left=.08,right=.73,bottom=.12,top=.88)
        for ext in ['png','pdf']:
            fig.savefig(out/f'figure2_bump_{model}.{ext}',
                        dpi=280 if ext=='png' else None,
                        bbox_inches='tight',facecolor='white')
        plt.close(fig)
    print('PASS | 1650 original-feature importance rows, 25 splits per combination')
    print('Created separate readable figures for LR and RF in',out)

if __name__=='__main__': main()
