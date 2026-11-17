#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Step 10: immutable-reference four-method manuscript tables and figures.

PROVISIONAL; ADVISOR APPROVAL PENDING.

Preflight is read-only. Run reads audited Stage 3/5/6-seedfix/7/8/9 files,
125 SHA256-verified saved LR classifiers and the original Stage 3 reporting
references; it NEVER refits any model, imputer or SHAP. All results are
aggregate. It writes only to a NEW output/stage4_mice/ child directory.

MICE LR coefficient estimand: within each split, average coefficients from
five fitted LR draws, then compute mean and sample SD of the 25 split means.
This is descriptive; coefficients remain in their respective fitted standard-
scaler units (not harmonized raw-unit coefficients).

Complete-case correlation/VIF reference is preserved exactly. A new Pearson
training-set correlation from Step 8 is shown separately; *no* MICE VIF is
invented or inferred from aggregate data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

SCHEMA = 'sina-stage4-step10-v1'
METHODS = ('CCA', 'Median/Mode', 'KNN k=5', 'MICE m=5')
LEGACY = METHODS[:3]
MODELS = ('LR', 'RF')
SPLITS = set(range(1, 26))
FEATURES = (
    'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDFMPIR',
    'BMXBMI', 'BMXWAIST', 'BP_SYS_MEAN', 'BP_DIA_MEAN', 'SMQ020', 'PAQ605'
)
NUMERIC = ('RIDAGEYR','INDFMPIR','BMXBMI','BMXWAIST','BP_SYS_MEAN','BP_DIA_MEAN')
PAIRS = tuple(combinations(METHODS, 2))

# Pinned known-good aggregate/reference files from independently audited runs.
PIN = {
 'stage3_cv': 'dac7dee1d500c04c0b8d970598111087a06d1e66735c891770edecd352321e78',
 'stage3_importance': 'f7f3121af57eb80b14a812ff7828bf38830b903b54d014664092585860f17418',
 'stage3_stability': 'aba9994e692836dff87a8bbc2afc08c466eaade5f9b25f3396c0bd3a4d61bde6',
 'stage5_cv': '4a607ddad657e35c3db5e0cdaeecbf156e07adfbd839d8ea4e6c9cbb409fda76',
 'stage6_importance': '5dff5c7167c48169f3cbc780da204be0b16e68ee427b68d13d38da1e59d3352e',
 'stage7_stability': '894df3061128440afac6f8a9659f45135f5a5a794bb78e58195c8f3672b4fa2f',
 'step9_results': '13c758e266d019e8e2142c096d417acc50f37364205800134a952ddd57a2b3c2',
 'legacy_performance': '4913f202bf76ce1053c8632d4f6066bb4c812fabb7b4ad13e42d26e1396812af',
 'legacy_stability': '4a3034413ae36993a1ca5e57cc949ece3320603153a7761c11e47aca233aa703',
 'legacy_importance_mean': 'eb96bd02b34790e8ea01995919d087d6516a6dc7418d17340e3e351042fd4e05',
 'legacy_top5': '1015bea99742839c66de44b55f22acfefd55f513707d83ff55eae494ffdada8a',
 'legacy_lr_coeff': '897409b03f557f010498d2f492d34bca4bc8d10ce0546aeed58156ebfa98c33f',
 'legacy_corr': '3ab2ac55ff72743243f8344b7e903e80fab8f3d667f6ae8c2ac5b7f8b49ac12e',
 'legacy_vif': 'bb7e152677523c759377501002f45918aa9e83389de366114dd37f46ec92e70e',
 'step8_run_config': 'f7bca6ae784004f2509396e344c38182d92757550209c634754f0e187ee56b92',
}
MANIFEST_PIN = {
 'step5_aggregate_sha256.json': '6e2daaa7cc25dafc139154094438b0b1a94733aec1dc61b715e09958dcd48e5f',
 'step6_aggregate_sha256.json': 'cb3c9077a5471e5caa4e9a51d79bdd583e0daf476da52c8a8d40794f407ca31e',
 'step7_aggregate_sha256.json': '73061e16170d46f0ea2e5b2f74dde85db34fa19c2a778d7f8dc85a3e21cf8379',
 'step8_aggregate_sha256.json': '61ae6209992af4384f89d06d1408d89287c2c3eb6c78a68166e759768a664822',
 'step9_aggregate_sha256.json': 'fea043e889721ed0d769f221a2735976f569ee16591ad59de700c35f0727b423',
}


def stop(msg):
    raise RuntimeError('STOP: ' + msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require_file(path, expected=None):
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        stop('missing/symlinked input: ' + str(p))
    digest = sha(p)
    if expected and digest != expected:
        stop('SHA256 mismatch: ' + str(p) + ' got=' + digest + ' expected=' + expected)
    return digest


def manifest_check(directory, name):
    p = directory / name
    require_file(p)
    record = json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(record, dict) or not record:
        stop('bad aggregate manifest: ' + str(p))
    for rel, expected in record.items():
        if not isinstance(rel, str) or not isinstance(expected, str) or len(expected) != 64:
            stop('invalid manifest entry in ' + str(p))
        target = directory / rel
        if target.resolve().parent != directory.resolve():
            stop('aggregate manifest escapes source directory')
        require_file(target, expected)
    return sha(p)


def read_csv(path):
    require_file(path)
    df = pd.read_csv(path)
    if not len(df) or df.columns.has_duplicates:
        stop('empty/malformed CSV: ' + str(path))
    return df


def keys_exact(df, columns, expected_count):
    if len(df) != expected_count or df.duplicated(list(columns)).any():
        stop('unexpected row count/duplicate keys: ' + '/'.join(columns))


def finite(df, columns):
    if not set(columns).issubset(df.columns):
        stop('missing numeric columns: ' + str(set(columns)-set(df.columns)))
    if not np.isfinite(df[list(columns)].to_numpy(dtype=float)).all():
        stop('nonfinite numerical source values: ' + str(columns))


def numeric_equal(a,b,where,atol=1e-12):
    if len(a) != len(b) or not np.allclose(np.asarray(a,dtype=float), np.asarray(b,dtype=float), atol=atol,rtol=0):
        stop('historical value changed: '+where)


def check_importance(df, methods):
    expected = 25*2*len(methods)*11
    keys_exact(df, ('split_id','model','method','feature'), expected)
    if set(df['model']) != set(MODELS) or set(df['method']) != set(methods) or set(df['feature']) != set(FEATURES):
        stop('wrong model/method/feature importance labels')
    if set(df['split_id']) != SPLITS:
        stop('missing importance splits')
    finite(df, ('split_id','mean_abs_shap','rank','mean_shap'))
    if (df['mean_abs_shap'] < 0).any():
        stop('negative mean_abs_shap')
    for (split,model,method),grp in df.groupby(['split_id','model','method']):
        if len(grp) != 11 or sorted(grp['rank'].tolist()) != list(range(1,12)):
            stop(f'wrong 11-feature ranks split={split} {model} {method}')
        exp = 'log_odds' if model == 'LR' else 'positive_class_probability'
        if not grp['shap_scale'].eq(exp).all():
            stop('wrong SHAP scale')
        # Top-five rank set must correspond to top-five numerical importances.
        vals = grp.sort_values('rank')['mean_abs_shap'].to_numpy()
        if np.any(vals[:-1] + 1e-12 < vals[1:]):
            stop('importance ranks disagree with mean absolute SHAP')


def stage3_numeric_reference(st3, original):
    # Historical 150 split/model/method metric rows pinned by the original SHA.
    keys_exact(st3, ('split_id','model','method'), 150)
    if set(st3['method']) != set(LEGACY) or set(st3['model']) != set(MODELS) or set(st3['split_id']) != SPLITS:
        stop('unexpected Stage 3 performance identities')
    finite(st3, ('roc_auc','brier_score','train_n','test_n'))
    for (_, _),g in st3.groupby(['model','method']):
        if len(g) != 25:stop('Stage 3 performance missing split')
    keys_exact(original, ('model','method'), 6)
    if set(original['method']) != set(LEGACY):stop('legacy performance methods invalid')
    for _,row in original.iterrows():
        g=st3[(st3.model==row['model'])&(st3.method==row['method'])]
        for output_col,input_col in [('mean_roc_auc','roc_auc'),('mean_brier_score','brier_score'),('mean_train_n','train_n'),('mean_test_n','test_n')]:
            numeric_equal([row[output_col]],[g[input_col].mean()],f'performance {row.model}/{row.method}/{output_col}')
        for output_col,input_col in [('sd_roc_auc','roc_auc'),('sd_brier_score','brier_score')]:
            numeric_equal([row[output_col]],[g[input_col].std(ddof=1)],f'performance {row.model}/{row.method}/{output_col}')
        if int(row['n_splits']) != 25:stop('wrong original n_splits')


def verify_stage5_lr_models(stage5, step8_config, mice_cv):
    pinned = step8_config.get('source',{}).get('stage5_split_checkpoint_markers_sha256',{})
    if len(pinned) != 25:stop('Step 8 lacks 25 pinned Stage 5 checkpoints')
    if set(pinned) != {f'split_{s:03d}/split_complete.json' for s in SPLITS}:
        stop('Step 8 checkpoint identities differ')
    model_paths = {}
    for split in range(1, 26):
        rel = f'split_{split:03d}/split_complete.json'
        marker_path = stage5 / rel
        require_file(marker_path, pinned[rel])
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        if marker.get('split_id') != split or len(marker.get('sha256',{}))!=18:
            stop('invalid Stage 5 split checkpoint schema '+rel)
        hashes = marker['sha256']
        for draw in range(1,6):
            model_rel = f'models/LR_draw{draw:02d}_classifier.joblib'
            f = stage5 / f'split_{split:03d}' / model_rel
            if model_rel not in hashes:
                stop('missing LR model hash '+str(f))
            require_file(f, hashes[model_rel])
            model_paths[(split,draw)] = f
        # Pin saved split-level MICE performance to aggregate output.
        local_rel = 'mice_performance.csv'
        if local_rel not in hashes:stop('missing split performance hash')
        local = stage5/f'split_{split:03d}'/local_rel
        require_file(local,hashes[local_rel])
        frame=read_csv(local)
        if len(frame)!=2 or set(frame.model)!=set(MODELS):stop('invalid split performance rows')
        for model in MODELS:
            a=frame.loc[frame.model==model].iloc[0]
            b=mice_cv[(mice_cv.split_id==split)&(mice_cv.model==model)].iloc[0]
            for col in ('roc_auc','brier_score'):
                numeric_equal([a[col]],[b[col]],'saved split performance '+col, atol=1e-14)
            if str(a['best_params_json'])!=str(b['best_params_json']):
                stop('saved split best_params_json differs')
    if len(model_paths)!=125:
        stop('wrong number of LR fitted models')
    return model_paths


def validate_data(args):
    root = Path.cwd().resolve()
    stage3 = (root/args.stage3_dir).resolve()
    stage5 = (root/args.stage5_dir).resolve()
    stage6 = (root/args.seedfix_dir).resolve()
    stage7 = (root/args.stage7_dir).resolve()
    stage8 = (root/args.stage8_dir).resolve()
    stage9 = (root/args.stage9_dir).resolve()
    original = (root/args.legacy_report_dir).resolve()
    numeric = (root/args.legacy_numeric_dir).resolve()
    paths = {
      'stage3_cv':stage3/'cv_metrics_final.csv',
      'stage3_importance':stage3/'shap/shap_global_importance_final.csv',
      'stage3_stability':stage3/'shap/shap_stability_metrics_by_split_final.csv',
      'stage5_cv':stage5/'mice_cv_metrics_25_splits.csv',
      'stage6_importance':stage6/'mice_shap_global_importance_25_splits.csv',
      'stage7_stability':stage7/'shap_stability_metrics_4_methods_25_splits.csv',
      'step9_results':stage9/'step9_corrected_intervals_4_methods.csv',
      'legacy_performance':original/'table1_predictive_performance.csv',
      'legacy_stability':original/'table2_shap_stability.csv',
      'legacy_importance_mean':numeric/'shap_mean_abs_by_feature.csv',
      'legacy_top5':numeric/'shap_top5_frequency.csv',
      'legacy_lr_coeff':numeric/'lr_waist_bmi_coefficients.csv',
      'legacy_corr':numeric/'collinearity_correlation.csv',
      'legacy_vif':numeric/'collinearity_vif.csv',
    }
    for key,path in paths.items():require_file(path,PIN[key])
    require_file(stage8/'step8_run_config.json', PIN['step8_run_config'])
    for dirname,name in [
      (stage5,'step5_aggregate_sha256.json'),
      (stage6,'step6_aggregate_sha256.json'),
      (stage7,'step7_aggregate_sha256.json'),
      (stage8,'step8_aggregate_sha256.json'),
      (stage9,'step9_aggregate_sha256.json'),
    ]:
        require_file(dirname/name, MANIFEST_PIN[name])
        manifest_check(dirname,name)
    s5=json.loads((stage5/'step5_summary.json').read_text(encoding='utf-8'))
    s6=json.loads((stage6/'step6_summary.json').read_text(encoding='utf-8'))
    s7=json.loads((stage7/'step7_summary.json').read_text(encoding='utf-8'))
    s8=json.loads((stage8/'step8_summary.json').read_text(encoding='utf-8'))
    s9=json.loads((stage9/'step9_summary.json').read_text(encoding='utf-8'))
    c8=json.loads((stage8/'step8_run_config.json').read_text(encoding='utf-8'))
    c9=json.loads((stage9/'step9_run_config.json').read_text(encoding='utf-8'))
    if s5.get('completed_splits')!=25 or s5.get('metric_rows')!=50:stop('Stage 5 summary incomplete')
    if s6.get('seed_correction_schema')!='sina-stage4-step6-seedfix-v1' or s6.get('rf_draws_corrected_splits_2_to_25')!=120:stop('Stage 6 NOT seedfixed')
    if s7.get('n_rows')!=300 or s7.get('n_legacy_rows_exactly_preserved')!=150:stop('Stage 7 summary incomplete')
    if s8.get('n_split_draw_records')!=125 or s8.get('n_output_csv_rows')!=16:stop('Stage 8 summary incomplete')
    if s9.get('rows')!=48 or s9.get('performance_rows')!=24:stop('Stage 9 summary incomplete')
    if c8.get('source',{}).get('stage5_aggregate_manifest_sha256')!=sha(stage5/'step5_aggregate_sha256.json'):
        stop('Stage 8 Stage 5 manifest provenance mismatch')
    if c9.get('input_sha256',{}).get('shap_stability_metrics_4_methods_25_splits.csv')!=PIN['stage7_stability']:
        stop('Stage 9 Stage 7 provenance mismatch')
    if s7.get('source_corrected_stage6_manifest_sha256')!=sha(stage6/'step6_aggregate_sha256.json'):
        stop('Stage 7 corrected Stage 6 provenance mismatch')
    print('STEP 10 SOURCE PASS | Stage 3/5/6-SEEDFIX/7/8/9 manifests and original report pinned SHA256',flush=True)

    st3=read_csv(paths['stage3_cv']);s5cv=read_csv(paths['stage5_cv'])
    st3imp=read_csv(paths['stage3_importance']);s6imp=read_csv(paths['stage6_importance'])
    stability=read_csv(paths['stage7_stability']);inference=read_csv(paths['step9_results'])
    original_perf=read_csv(paths['legacy_performance']);original_stab=read_csv(paths['legacy_stability'])
    original_mean=read_csv(paths['legacy_importance_mean']);original_top5=read_csv(paths['legacy_top5'])
    original_coeff=read_csv(paths['legacy_lr_coeff'])
    stage3_numeric_reference(st3,original_perf)
    keys_exact(s5cv,('split_id','model','method'),50)
    if set(s5cv['split_id'])!=SPLITS or set(s5cv.model)!=set(MODELS) or set(s5cv.method)!={'MICE m=5'}:
        stop('Stage 5 performance not 25x2 MICE')
    finite(s5cv,('roc_auc','brier_score','n_train','n_test_common'))
    if not s5cv.n_imputations.eq(5).all():stop('Stage 5 not 5 MICE draws')
    check_importance(st3imp,LEGACY);check_importance(s6imp,('MICE m=5',))
    old=st3imp[['split_id','model','method','feature','mean_abs_shap','rank']]
    new=s6imp[['split_id','model','method','feature','mean_abs_shap','rank']]
    importance=pd.concat([old,new],ignore_index=True)
    if len(importance)!=2200:stop('incorrect combined SHAP importance rows')
    # Confirm historical per-feature SHAP mean and SD against original numeric report.
    keys_exact(original_mean,('model','method','feature'),66)
    keys_exact(original_top5,('model','method','feature'),66)
    keys_exact(original_coeff,('model','method','feature'),6)
    assert_top5=[];assert_shap=[]
    agg=importance.groupby(['model','method','feature'],as_index=False).agg(
      mean_abs_shap=('mean_abs_shap','mean'),sd_abs_shap=('mean_abs_shap','std'),
      n_splits=('split_id','nunique'))
    top=importance.assign(in_top5=lambda d:d['rank']<=5).groupby(
      ['model','method','feature'],as_index=False).agg(n_splits_in_top5=('in_top5','sum'))
    for _,r in original_mean.iterrows():
        q=agg[(agg.model==r.model)&(agg.method==r.method)&(agg.feature==r.feature)]
        if len(q)!=1:stop('missing historical importance feature')
        for k in ('mean_abs_shap','sd_abs_shap'):
            numeric_equal([q.iloc[0][k]],[r[k]],f'legacy importance {r.model}/{r.method}/{r.feature}/{k}',atol=1e-10)
    for _,r in original_top5.iterrows():
        q=top[(top.model==r.model)&(top.method==r.method)&(top.feature==r.feature)]
        if len(q)!=1 or int(q.iloc[0].n_splits_in_top5)!=int(r.n_splits_in_top5):
            stop('historical top5 counts changed')
    if not (top.groupby(['model','method'])['n_splits_in_top5'].sum()==125).all():
        stop('incorrect top-five frequency sums')
    keys_exact(stability,('split_id','model','method_a','method_b'),300)
    if set(stability.split_id)!=SPLITS or set(stability.model)!=set(MODELS):stop('SHAP stability incomplete')
    if set(zip(stability.method_a,stability.method_b))!=set(PAIRS):stop('SHAP stability six pairs invalid')
    finite(stability,('spearman_rank_correlation','top5_jaccard','mse_shap','mae_shap'))
    keys_exact(original_stab,('model','method_a','method_b'),6)
    for _,r in original_stab.iterrows():
        q=stability[(stability.model==r.model)&(stability.method_a==r.method_a)&(stability.method_b==r.method_b)]
        if len(q)!=25:stop('missing original SHAP stability pair')
        for src,slug in [('spearman_rank_correlation','spearman'),('top5_jaccard','top5_jaccard'),('mse_shap','mse_shap'),('mae_shap','mae_shap')]:
            numeric_equal([q[src].mean()],[r['mean_'+slug]],'legacy stability mean '+slug)
            numeric_equal([q[src].std(ddof=1)],[r['sd_'+slug]],'legacy stability SD '+slug)
    keys_exact(inference,('metric_family','model','method_a','method_b'),48)
    if set(inference.metric_family)!={'auc','brier','spearman','jaccard'}:
        stop('unexpected Stage 9 metric family')
    # Stage 9 is the authoritative corrected-NB/Holm table, never recompute its values here.
    if len(inference[inference.metric_family.isin(['auc','brier'])])!=24:
        stop('Stage 9 performance table count differs')
    print('STEP 10 HISTORICAL PASS | original 6 performance + 6 stability + 66 importance + 66 top5 + 6 LR coefficient rows preserved',flush=True)
    print('STEP 10 SHAP PASS | 2,200 feature importance records; 300 paired SHAP rows; 4 methods x 2 models',flush=True)
    model_paths=verify_stage5_lr_models(stage5,c8,s5cv)
    print('STEP 10 MODEL PASS | 125/125 fitted LR classifiers SHA256 verified; no model loading during preflight',flush=True)
    return {
        'root':root,'paths':paths,'dirs':{'stage3':stage3,'stage5':stage5,'stage6':stage6,'stage7':stage7,'stage8':stage8,'stage9':stage9},
        'st3':st3,'s5cv':s5cv,'importance':importance,'stability':stability,'inference':inference,
        'original_perf':original_perf,'original_stab':original_stab,'original_mean':original_mean,
        'original_top5':original_top5,'original_coeff':original_coeff,'model_paths':model_paths,
        'agg_importance':agg,'agg_top5':top,'step8_summary':s8,
    }


def summarize_lr_mice_models(state):
    # Load ONLY 125 checkpoint-hash-verified trusted Stage 5 sklearn joblibs.
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    result=[]
    for split in range(1,26):
        params=json.loads(state['s5cv'].query('split_id == @split and model == "LR"').iloc[0]['best_params_json'])
        expected_c=float(params['C'])
        for draw in range(1,6):
            path=state['model_paths'][(split,draw)]
            model=joblib.load(path) # Trusted, SHA256-verified, local model only.
            if not isinstance(model,Pipeline) or list(model.named_steps)!=['preprocess','classifier']:
                stop('noncanonical fitted LR pipeline '+str(path))
            pre=model.named_steps['preprocess']; lr=model.named_steps['classifier']
            if not isinstance(pre,ColumnTransformer) or not isinstance(lr,LogisticRegression):
                stop('invalid saved LR estimator '+str(path))
            if not np.isclose(float(lr.C),expected_c,rtol=1e-12,atol=1e-12):
                stop('LR C mismatch at '+str(path))
            if list(lr.classes_)!=[0,1]:stop('LR class ordering mismatch')
            num_cols=next((tuple(cols) for name,_,cols in pre.transformers_ if name=='numeric'),None)
            if num_cols!=NUMERIC:stop('numeric features different from Stage 3 design')
            scaler=pre.named_transformers_.get('numeric')
            if not isinstance(scaler,StandardScaler):
                stop('MICE LR must use CCA-style StandardScaler fitted on imputed training data')
            if len(scaler.scale_)!=len(NUMERIC) or not np.isfinite(scaler.scale_).all() or not np.all(scaler.scale_>0):
                stop('invalid saved StandardScaler')
            names=list(pre.get_feature_names_out())
            coef=np.asarray(lr.coef_,dtype=float)
            if coef.shape!=(1,len(names)) or not np.isfinite(coef).all():
                stop('bad saved LR coefficient shape/values')
            for feature in ('BMXBMI','BMXWAIST'):
                name='numeric__'+feature
                if names.count(name)!=1:stop('target feature absent or duplicated: '+name)
                result.append({'split_id':split,'draw':draw,'feature':feature,'coef':float(coef[0,names.index(name)])})
    frame=pd.DataFrame(result)
    keys_exact(frame,('split_id','draw','feature'),250)
    pooled=frame.groupby(['split_id','feature'],as_index=False)['coef'].mean()
    if len(pooled)!=50:stop('incorrect pooled LR coefficient count')
    rows=[]
    for feature in ('BMXBMI','BMXWAIST'):
        g=pooled.loc[pooled.feature==feature,'coef']
        if len(g)!=25:stop('missing LR pooled split coefficient')
        rows.append({'model':'LR','method':'MICE m=5','feature':feature,
          'mean_coefficient_standardized':float(g.mean()),
          'sd_coefficient_standardized':float(g.std(ddof=1)),
          'n_splits_used':25})
    return pd.DataFrame(rows)


def tables_from_data(state,mice_coeff):
    perf=state['original_perf'].copy()
    add=[]
    for model in MODELS:
        g=state['s5cv'].loc[state['s5cv'].model==model]
        if len(g)!=25:stop('MICE performance split count incorrect')
        mean_auc=float(g.roc_auc.mean());sd_auc=float(g.roc_auc.std(ddof=1))
        mean_brier=float(g.brier_score.mean());sd_brier=float(g.brier_score.std(ddof=1))
        add.append({'model':model,'method':'MICE m=5','n_splits':25,
          'mean_roc_auc':mean_auc,'sd_roc_auc':sd_auc,
          'roc_auc_mean_sd':f'{mean_auc:.4f} ± {sd_auc:.4f}',
          'mean_brier_score':mean_brier,'sd_brier_score':sd_brier,
          'brier_mean_sd':f'{mean_brier:.4f} ± {sd_brier:.4f}',
          'mean_train_n':float(g.n_train.mean()),'mean_test_n':float(g.n_test_common.mean())})
    perf=pd.concat([perf,pd.DataFrame(add)[perf.columns]],ignore_index=True)
    keys_exact(perf,('model','method'),8)

    stab=state['original_stab'].copy()
    rows=[]
    for model in MODELS:
        for pa in LEGACY:
            pb='MICE m=5'
            g=state['stability'].loc[(state['stability'].model==model)&(state['stability'].method_a==pa)&(state['stability'].method_b==pb)]
            if len(g)!=25:stop('missing MICE comparison in stability')
            d={'model':model,'method_a':pa,'method_b':pb,'n_splits':25,
               'shap_scale':'log_odds' if model=='LR' else 'positive_class_probability'}
            for src,slug in [('spearman_rank_correlation','spearman'),('top5_jaccard','top5_jaccard'),('mse_shap','mse_shap'),('mae_shap','mae_shap')]:
                m=float(g[src].mean());s=float(g[src].std(ddof=1))
                d['mean_'+slug]=m;d['sd_'+slug]=s;d[slug+'_mean_sd']=f'{m:.4g} ± {s:.4g}'
            rows.append(d)
    stab=pd.concat([stab,pd.DataFrame(rows)[stab.columns]],ignore_index=True)
    keys_exact(stab,('model','method_a','method_b'),12)

    imp=state['original_mean'].copy()
    mice=state['agg_importance'].loc[state['agg_importance'].method=='MICE m=5',list(imp.columns)]
    imp=pd.concat([imp,mice],ignore_index=True)
    keys_exact(imp,('model','method','feature'),88)
    top=state['original_top5'].copy()
    mice_top=state['agg_top5'].loc[state['agg_top5'].method=='MICE m=5',list(top.columns)]
    top=pd.concat([top,mice_top],ignore_index=True)
    keys_exact(top,('model','method','feature'),88)
    coefs=pd.concat([state['original_coeff'],mice_coeff[list(state['original_coeff'].columns)]],ignore_index=True)
    keys_exact(coefs,('model','method','feature'),8)

    corr=read_csv(state['paths']['legacy_corr']).copy()
    ref_r=float(corr.loc[(corr.method=='Pearson') & (corr.feature_1=='BMXWAIST') & (corr.feature_2=='BMXBMI'),'correlation'].iloc[0])
    mice_r=float(state['step8_summary']['waist_bmi_correlation']['mean_pearson_r'])
    rsd=float(state['step8_summary']['waist_bmi_correlation']['sd_pearson_r_across_125'])
    context=pd.DataFrame([
      {'population':'complete_case_reference','method':'Pearson','feature_1':'BMXWAIST','feature_2':'BMXBMI',
       'mean_correlation':ref_r,'sd_descriptive':'','n_split_draw_records':0,
       'note':'Original 4031 complete-case subjects; original aggregate Pearson correlation'},
      {'population':'MICE_imputed_training','method':'Pearson','feature_1':'BMXWAIST','feature_2':'BMXBMI',
       'mean_correlation':mice_r,'sd_descriptive':rsd,'n_split_draw_records':125,
       'note':'Descriptive mean/SD across dependent 25 splits x 5 imputation draws; not pooled independent inference'}
    ])
    return {
      'table1_predictive_performance_4_methods.csv':perf,
      'table2_paired_NB_Holm_4_methods.csv':state['inference'].loc[state['inference'].metric_family.isin(['auc','brier'])].copy(),
      'table3_shap_stability_4_methods.csv':stab,
      'table4_shap_mean_abs_by_feature_4_methods.csv':imp,
      'table5_shap_top5_frequency_4_methods.csv':top,
      'table6_lr_waist_bmi_coefficients_4_methods.csv':coefs,
      'table7_waist_bmi_correlation_context.csv':context,
      'table8_collinearity_vif_complete_case_reference.csv':read_csv(state['paths']['legacy_vif']),
      'table9_collinearity_correlation_complete_case_reference.csv':corr,
    }


def build_figures(out,importance):
    # Import plotting only during --mode run; preflight never draws/creates files.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    for model in MODELS:
        mm=importance[importance.model==model]
        agg=mm.groupby(['method','feature'],as_index=False).agg(
             mean_abs=('mean_abs_shap','mean'),sd_abs=('mean_abs_shap','std'),mean_rank=('rank','mean'))
        # Figure 1: the same eleven original predictors across four methods.
        fig,ax=plt.subplots(figsize=(16.4,7.2))
        x=np.arange(11);width=0.20
        for j,method in enumerate(METHODS):
            vals=agg[agg.method==method].set_index('feature').reindex(FEATURES)
            if vals[['mean_abs','sd_abs']].isna().any().any():stop('incomplete plot 1 data')
            ax.bar(x+(j-1.5)*width,vals['mean_abs'].to_numpy(),width,
                   yerr=vals['sd_abs'].to_numpy(),capsize=2.0,label=method)
        ax.set_xticks(x,FEATURES,rotation=43,ha='right',fontsize=10)
        ax.set_title(f'Figure 1 — Global SHAP importance by missing-data strategy ({model})',fontsize=14)
        ax.set_ylabel('Mean absolute SHAP value (mean ± SD across 25 splits)')
        ax.set_xlabel('Original feature (11 predictors)')
        ax.grid(axis='y',alpha=.25)
        ax.set_axisbelow(True)
        ax.legend(title='Missing-data strategy',bbox_to_anchor=(1.01,0.5),loc='center left',frameon=False)
        fig.subplots_adjust(left=.08,right=.78,bottom=.24,top=.91)
        for ext in ('png','pdf'):
            fig.savefig(out/f'figure1_mean_abs_shap_{model}.{ext}',
                        dpi=300 if ext=='png' else None,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        # Figure 2: all feature labels are only in the exterior legend.
        fig,ax=plt.subplots(figsize=(12.8,7.2))
        markers=['o','s','^','D','v','P','X','>','<','h','*']
        for i,feature in enumerate(FEATURES):
            vals=agg[agg.feature==feature].set_index('method').reindex(METHODS)
            y=vals['mean_rank'].to_numpy(dtype=float)
            if not np.isfinite(y).all():stop('incomplete bump plot mean ranks')
            ax.plot(range(4),y,marker=markers[i],markersize=6,linewidth=1.8,label=feature)
        ax.set_xticks(range(4),METHODS,fontsize=10)
        ax.set_xlim(-.15,3.15)
        ax.set_yticks(range(1,12));ax.set_ylim(11.4,.6)
        ax.grid(axis='y',alpha=.3)
        ax.set_ylabel('Mean feature rank across 25 splits (1 = most important)')
        ax.set_title(f'Figure 2 — SHAP feature-rank stability across four strategies ({model})',fontsize=14)
        ax.legend(title='Original predictor',loc='center left',bbox_to_anchor=(1.01,.5),fontsize=10,frameon=False)
        fig.subplots_adjust(left=.10,right=.72,bottom=.13,top=.9)
        for ext in ('png','pdf'):
            fig.savefig(out/f'figure2_feature_rank_bump_{model}.{ext}',
                        dpi=300 if ext=='png' else None,bbox_inches='tight',facecolor='white')
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','run'),required=True)
    p.add_argument('--stage3-dir',default='output/stage3_final')
    p.add_argument('--legacy-report-dir',default='output/stage3_reporting_v1')
    p.add_argument('--legacy-numeric-dir',default='advisor_remaining_csv')
    p.add_argument('--stage5-dir',required=True)
    p.add_argument('--seedfix-dir',required=True)
    p.add_argument('--stage7-dir',required=True)
    p.add_argument('--stage8-dir',required=True)
    p.add_argument('--stage9-dir',required=True)
    p.add_argument('--out',help='New and nonexistent directory inside output/stage4_mice')
    args=p.parse_args()
    if args.mode=='run' and not args.out:stop('--out is required for --mode run')
    state=validate_data(args)
    if args.mode=='preflight':
        print('STEP 10 PREFLIGHT PASS | read-only; 9 tables + 4 PNG (300 dpi) + 4 PDF ready; 125 LR models for coefficient pooling; no output written',flush=True)
        return
    out=(state['root']/args.out).resolve()
    stage4=(state['root']/'output/stage4_mice').resolve()
    if not out.is_relative_to(stage4) or out==stage4:
        stop('new report output must be a child of output/stage4_mice/')
    if out.exists():stop('refusing to overwrite existing output '+str(out))
    if not out.parent.is_dir():stop('output parent directory does not exist '+str(out.parent))
    if out.parent.is_symlink():stop('symlinked output parent forbidden')
    print('STEP 10 COEFFICIENTS | reading 125 trusted Stage 5 LR classifiers; no retraining',flush=True)
    mice_coeff=summarize_lr_mice_models(state)
    tables=tables_from_data(state,mice_coeff)
    # Final output is only promoted into place after every table and figure passes.
    working=out.parent/(out.name+'.tmp_'+str(os.getpid()))
    if working.exists():stop('temporary target exists '+str(working))
    working.mkdir(exist_ok=False)
    try:
        for filename,df in tables.items():
            if filename in ('table8_collinearity_vif_complete_case_reference.csv', 'table9_collinearity_correlation_complete_case_reference.csv'):
                src=state['paths']['legacy_vif' if filename.startswith('table8') else 'legacy_corr']
                shutil.copyfile(src,working/filename)  # preserve original bytes exactly
            elif filename=='table2_paired_NB_Holm_4_methods.csv':
                # Stage 9 file has 48 rows; emit only inferential 24 rows, preserving all numeric values.
                df.to_csv(working/filename,index=False,float_format='%.17g')
            else:df.to_csv(working/filename,index=False,float_format='%.17g')
            reread=read_csv(working/filename)
            if len(reread)!=len(df):stop('output table failed CSV roundtrip '+filename)
        build_figures(working,state['importance'])
        figs=sorted(working.glob('figure*.png'))
        if len(figs)!=4 or len(list(working.glob('figure*.pdf')))!=4:
            stop('expected four PNG and four PDF figures')
        from PIL import Image
        dims={}
        for fig in figs:
            with Image.open(fig) as image:
                dpi=image.info.get('dpi',(0,0))
                if not 299<=dpi[0]<=301 or not 299<=dpi[1]<=301:
                    stop('PNG not 300 dpi '+fig.name+' '+str(dpi))
                if image.width<2000 or image.height<1000:stop('PNG unexpectedly small '+fig.name)
                dims[fig.name]={'pixels':[image.width,image.height],'dpi':[float(dpi[0]),float(dpi[1])]}
        refs={k:sha(v) for k,v in state['paths'].items()}
        refs.update({'stage5_manifest':sha(state['dirs']['stage5']/'step5_aggregate_sha256.json'),
            'stage6_manifest':sha(state['dirs']['stage6']/'step6_aggregate_sha256.json'),
            'stage7_manifest':sha(state['dirs']['stage7']/'step7_aggregate_sha256.json'),
            'stage8_manifest':sha(state['dirs']['stage8']/'step8_aggregate_sha256.json'),
            'stage9_manifest':sha(state['dirs']['stage9']/'step9_aggregate_sha256.json')})
        summary={
          'schema':SCHEMA,'advisor_status':'PENDING; PROVISIONAL',
          'n_methods':4,'n_models':2,'n_splits':25,'n_original_features':11,
          'n_mice_lr_models_loaded':125,
          'coefficient_rule':'mean 5 LR draw coefficients per split, then mean and sample SD of 25 split averages; fitted StandardScaler units',
          'original_rows_preserved':{'performance':6,'shap_stability':6,'global_importance':66,'top5':66,'lr_coefficients':6},
          'output_tables':{k:int(len(v)) for k,v in tables.items()},
          'figure_png_300dpi':dims,
          'legacy_collinearity_note':'Complete-case VIF/correlation are reference data, not MICE-specific VIF; MICE training Pearson r comes from Step 8 and is descriptive.',
          'shap_scale_note':'LR log_odds; RF positive_class_probability; SHAP magnitude not compared across models.',
          'uncertainty_note':'SHAP plots show split SD, not a confidence interval; Stage 9 Holm applies only to AUC and Brier.',
          'data_privacy':'Only aggregate CSV and figures; no participant-level values exported.',
          'source_sha256':refs,
          'created_utc':datetime.now(timezone.utc).isoformat(),
        }
        (working/'step10_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        config={'schema':SCHEMA,'source_dirs':{k:str(v) for k,v in state['dirs'].items()},
                'legacy_reference_dirs':{'report':str((state['root']/args.legacy_report_dir).resolve()),
                                         'numeric':str((state['root']/args.legacy_numeric_dir).resolve())},
                'script_sha256':sha(Path(__file__).resolve()),'advisor_status':'PENDING; PROVISIONAL'}
        (working/'step10_run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        hashes={p.name:sha(p) for p in sorted(working.iterdir()) if p.is_file()}
        (working/'step10_aggregate_sha256.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        # Independent final checksum recheck, then atomic output promotion.
        for name,digest in hashes.items():require_file(working/name,digest)
        working.rename(out)
    except BaseException:
        # Avoid promotion; preserve a failed new temp directory for inspection.
        print('STEP 10 ERROR | no final output promoted; inspect temporary directory: '+str(working),file=sys.stderr)
        raise
    print('STEP 10 PASS | 9 tables + 4 PNG (300 dpi) + 4 PDF; legacy historical numbers preserved; 125 MICE LR models; SHA256 PASS',flush=True)
    print('STEP 10 OUTPUT | '+str(out),flush=True)


if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError,KeyError,TypeError,FileNotFoundError) as e:
        print('STEP 10 FAIL | '+str(e),file=sys.stderr,flush=True)
        sys.exit(2)
