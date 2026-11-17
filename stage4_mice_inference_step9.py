#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINA Step 9. Paired Nadeau-Bengio corrected intervals and Holm.

PROVISIONAL; ADVISOR APPROVAL PENDING. No model training or SHAP computation.
Preflight is read-only; run writes a NEW output directory only.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
import scipy.stats as stats
sys.dont_write_bytecode = True

METHODS = ('CCA','Median/Mode','KNN k=5','MICE m=5')
MODELS = ('LR','RF')
SPLITS = tuple(range(1,26))
PAIRS = tuple(combinations(METHODS,2))
EXPECTED_STAGE3 = 'dac7dee1d500c04c0b8d970598111087a06d1e66735c891770edecd352321e78'
EXPECTED_STAGE5 = '4a607ddad657e35c3db5e0cdaeecbf156e07adfbd839d8ea4e6c9cbb409fda76'
EXPECTED_STAGE7 = '894df3061128440afac6f8a9659f45135f5a5a794bb78e58195c8f3672b4fa2f'
FIELDS = ('metric_family','model','method_a','method_b','difference_definition','n_paired_splits','mean_difference','sd_difference','mean_n_train_smaller','mean_n_test','correction_factor','sigma_corr','df','ci95_lower','ci95_upper','t_stat','p_raw','p_holm','holm_reject_0_05','inference_scope')

def stop(msg): raise RuntimeError('STOP: '+msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def csv_rows(p):
 if not p.is_file():stop('missing file '+str(p))
 with p.open(newline='',encoding='utf-8-sig') as f:
  r=csv.DictReader(f);rows=list(r)
  if not rows or any(None in x or any(v is None for v in x.values()) for x in rows):stop('invalid CSV '+str(p))
  return rows

def checked_float(v, label):
 try:n=float(v)
 except (TypeError,ValueError):stop('invalid '+label)
 if not math.isfinite(n):stop('nonfinite '+label)
 return n

def checked_int(v, label):
 n=checked_float(v,label)
 if not n.is_integer():stop('noninteger '+label)
 return int(n)

def source_check(args):
 stage3=Path(args.stage3_cv).expanduser().resolve()
 stage5=Path(args.stage5_dir).expanduser().resolve()
 stage7=Path(args.stage7_dir).expanduser().resolve()
 files=[stage3,stage5/'mice_cv_metrics_25_splits.csv',stage7/'shap_stability_metrics_4_methods_25_splits.csv']
 expected=[EXPECTED_STAGE3,EXPECTED_STAGE5,EXPECTED_STAGE7]
 for p,e in zip(files,expected):
  if not p.is_file() or p.is_symlink():stop('missing/symlinked '+str(p))
  got=sha(p)
  if got!=e:stop('unexpected input SHA256 '+str(p)+' observed='+got)
 for dirname,manifest_name,required in [
  (stage5,'step5_aggregate_sha256.json','mice_cv_metrics_25_splits.csv'),
  (stage7,'step7_aggregate_sha256.json','shap_stability_metrics_4_methods_25_splits.csv')]:
  manifest=dirname/manifest_name
  if not manifest.is_file() or manifest.is_symlink():stop('missing manifest '+str(manifest))
  record=json.loads(manifest.read_text())
  if record.get(required)!=sha(dirname/required):stop('aggregate manifest mismatch '+required)
  for name,hs in record.items():
   p=dirname/name
   if not p.is_file() or p.is_symlink() or sha(p)!=hs:stop('source manifest file mismatch '+str(p))
 s5=json.loads((stage5/'step5_summary.json').read_text())
 s7=json.loads((stage7/'step7_summary.json').read_text())
 if s5.get('completed_splits')!=25 or s5.get('metric_rows')!=50:stop('Stage 5 summary not complete')
 if s7.get('n_rows')!=300 or s7.get('n_legacy_rows_exactly_preserved')!=150 or not s7.get('all_stage6_corrected_rf_seeds_verified'):stop('Stage 7 summary not complete')
 print('STEP 9 SOURCE PASS | Stage 3 pinned SHA256; Stage 5/7 aggregate manifests verified',flush=True)
 return files, (csv_rows(files[0]),csv_rows(files[1]),csv_rows(files[2]))

def canonical(pair):return (pair[0],pair[1])
def validate_rows(groups):
 a,b,c=groups
 if (len(a),len(b),len(c))!=(150,50,300):stop('unexpected input row counts')
 performance={}; stab={}; split_meta={}
 for rows,is_mice in ((a,False),(b,True)):
  for row in rows:
   s=checked_int(row['split_id'],'split_id');m=row['model'];method=row['method']
   if s not in SPLITS or m not in MODELS or method not in METHODS or (method=='MICE m=5')!=is_mice:stop('unexpected performance identity')
   key=(s,m,method)
   if key in performance:stop('duplicated performance row')
   rep=checked_int(row['repeat'],'repeat');fold=checked_int(row['fold'],'fold')
   # Repeated 5-fold CV: split number implies repeat/fold.
   if (rep,fold)!=((s-1)//5+1,(s-1)%5+1):stop('unexpected repeat/fold')
   train=checked_int(row['n_train' if is_mice else 'train_n'],'n_train')
   test=checked_int(row['n_test_common' if is_mice else 'test_n'],'n_test')
   if train<=test or test<=0:stop('invalid train/test sizes')
   if (s,m) in split_meta and split_meta[(s,m)]!=test:stop('mismatched paired test size')
   split_meta[(s,m)]=test
   auc=checked_float(row['roc_auc'],'AUC');brier=checked_float(row['brier_score'],'Brier')
   if not (0<=auc<=1 and 0<=brier<=1):stop('metric out of range')
   if is_mice and checked_int(row['n_imputations'],'n_imputations')!=5:stop('MICE draws not 5')
   performance[key]={'train':train,'test':test,'auc':auc,'brier':brier}
 for row in c:
  s=checked_int(row['split_id'],'stability split');model=row['model'];pa=row['method_a'];pb=row['method_b']
  if s not in SPLITS or model not in MODELS or (pa,pb) not in PAIRS:stop('invalid SHAP pair identity')
  key=(s,model,pa,pb)
  if key in stab:stop('duplicate SHAP pair')
  if checked_int(row['repeat'],'repeat')!=(s-1)//5+1 or checked_int(row['fold'],'fold')!=(s-1)%5+1:stop('SHAP fold misalignment')
  if checked_int(row['test_n'],'test_n')!=split_meta[(s,model)] or checked_int(row['feature_n'],'feature_n')!=11:stop('SHAP population/feature count mismatch')
  spearman=checked_float(row['spearman_rank_correlation'],'Spearman');jaccard=checked_float(row['top5_jaccard'],'Jaccard')
  if not (-1<=spearman<=1 and 0<=jaccard<=1):stop('SHAP metric out of range')
  expected_scale='log_odds' if model=='LR' else 'positive_class_probability'
  if row['shap_scale']!=expected_scale:stop('SHAP scale mismatch')
  stab[key]={'spearman':spearman,'jaccard':jaccard}
 if len(performance)!=200 or len(stab)!=300 or len(split_meta)!=50:stop('missing/duplicate split-model-method identity')
 if {k[0] for k in performance}!=set(SPLITS):stop('incomplete split coverage')
 print('STEP 9 ALIGNMENT PASS | 200 performance rows; 300 SHAP pairs; 25 aligned outer splits; 11 features',flush=True)
 return performance,stab

def row_values(metric,model,pa,pb,performance,stab):
 diffs=[];trains=[];tests=[]
 for s in SPLITS:
  x=performance[(s,model,pa)];y=performance[(s,model,pb)]
  if x['test']!=y['test']:stop('test mismatch')
  trains.append(min(x['train'],y['train']));tests.append(x['test'])
  if metric in ('auc','brier'):diffs.append(x[metric]-y[metric])
  else:diffs.append(stab[(s,model,pa,pb)][metric])
 return diffs,mean(trains),mean(tests)

def construct(metric,model,pa,pb,performance,stab):
 vals,train,test=row_values(metric,model,pa,pb,performance,stab)
 correction=1/25+test/train
 sd=stdev(vals);sigma=math.sqrt(correction)*sd
 m=mean(vals);tcrit=stats.t.ppf(0.975,24)
 low=m-tcrit*sigma;high=m+tcrit*sigma
 infer=metric in ('auc','brier')
 if infer:
  if sigma==0:
   tstat=0.0 if m==0 else math.copysign(math.inf,m)
   p=1.0 if m==0 else 0.0
  else:
   tstat=m/sigma;p=2*stats.t.sf(abs(tstat),24)
 else:tstat='';p=''
 return {'metric_family':metric,'model':model,'method_a':pa,'method_b':pb,
 'difference_definition':'method_a_minus_method_b' if infer else 'descriptive_mean_pairwise_similarity',
 'n_paired_splits':25,'mean_difference':m,'sd_difference':sd,
 'mean_n_train_smaller':train,'mean_n_test':test,'correction_factor':correction,
 'sigma_corr':sigma,'df':24,'ci95_lower':low,'ci95_upper':high,'t_stat':tstat,
 'p_raw':p,'p_holm':'','holm_reject_0_05':'','inference_scope':'inferential_NB_Holm' if infer else 'descriptive_NB_CI_only'}

def apply_holm(rows,metric):
 family=[r for r in rows if r['metric_family']==metric]
 if len(family)!=12:stop('Holm family is not 12')
 order=sorted(range(12),key=lambda i:family[i]['p_raw'])
 previous=0.0
 for j,i in enumerate(order):
  adjusted=min(1.0,max(previous,(12-j)*family[i]['p_raw']))
  family[i]['p_holm']=adjusted;family[i]['holm_reject_0_05']=adjusted<0.05
  previous=adjusted

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--mode',choices=['preflight','run'],required=True)
 ap.add_argument('--stage3-cv',required=True)
 ap.add_argument('--stage5-dir',required=True)
 ap.add_argument('--stage7-dir',required=True)
 ap.add_argument('--out')
 args=ap.parse_args()
 if args.mode=='run' and not args.out:stop('--out required for run')
 files,groups=source_check(args)
 performance,stab=validate_rows(groups)
 rows=[]
 for metric in ('auc','brier','spearman','jaccard'):
  for model in MODELS:
   for pa,pb in PAIRS:rows.append(construct(metric,model,pa,pb,performance,stab))
 for family in ('auc','brier'):apply_holm(rows,family)
 if len(rows)!=48:stop('not 48 inference rows')
 for r in rows:
  if not (math.isfinite(r['sigma_corr']) and r['ci95_lower']<=r['mean_difference']<=r['ci95_upper']):stop('invalid interval')
  if r['metric_family'] in ('spearman','jaccard') and any(r[z]!='' for z in ('t_stat','p_raw','p_holm','holm_reject_0_05')):stop('descriptive has p value')
 print('STEP 9 MATH PASS | 24 performance tests (12 AUC + 12 Brier), Holm separately; 24 descriptive Spearman/Jaccard intervals',flush=True)
 if args.mode=='preflight':
  print('STEP 9 PREFLIGHT PASS | read-only, 48-row reporting calculation ready; no models/SHAP/files written',flush=True);return
 out=Path(args.out).expanduser().resolve()
 if out.exists():stop('output already exists: '+str(out))
 # All validation above happens before creating the new output directory.
 out.mkdir(parents=True,exist_ok=False)
 table=out/'step9_corrected_intervals_4_methods.csv'
 with table.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
 summary={'schema':'sina-stage4-step9-v1','advisor_status':'PENDING; PROVISIONAL',
 'n_splits':25,'n_models':2,'n_pairs':6,'rows':48,'performance_rows':24,'descriptive_shap_rows':24,
 'auc_holm_family_size':12,'brier_holm_family_size':12,'degrees_of_freedom':24,
 'variance_formula':'(1/25 + mean_n_test/mean_n_train_smaller) * sample_variance_over_25_splits',
 'point_difference_rule':'method_a minus method_b; Brier negative favors method_a',
 'shap_CI_rule':'descriptive only; do not present hypothesis p-values',
 'cv_dependency_warning':'25 splits are repeated 5-fold CV; corrected t is approximate; no independent-sample interpretation',
 'input_sha256':{p.name:sha(p) for p in files},'created_utc':datetime.now(timezone.utc).isoformat()}
 (out/'step9_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 config={'schema':'sina-stage4-step9-v1','script_sha256':sha(Path(__file__).resolve()),
 'stage3_cv':str(files[0]),'stage5_dir':str(files[1].parent),'stage7_dir':str(files[2].parent),
 'input_sha256':summary['input_sha256']}
 (out/'step9_run_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 manifest={name:sha(out/name) for name in ('step9_corrected_intervals_4_methods.csv','step9_summary.json','step9_run_config.json')}
 (out/'step9_aggregate_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 print('STEP 9 PASS | 48 rows; AUC 12+ Brier 12 Holm; Spearman 12+ Jaccard 12 descriptive corrected CI; SHA256 PASS',flush=True)
 print('STEP 9 OUTPUT | '+str(out),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  print('STEP 9 FAIL | '+str(exc),file=sys.stderr,flush=True);sys.exit(1)
