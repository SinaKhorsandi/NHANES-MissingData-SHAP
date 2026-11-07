"""Independent audit reports for the NHANES 2017-2018 stage-one files.

Run after: python prepare_nhanes.py
Requires original raw_xpt/*.XPT and generated output/*.csv.
Produces data-quality reports only; does NOT train, delete, or impute.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw_xpt'
OUT=ROOT/'output'


def sas(file: str, cols: list[str]) -> pd.DataFrame:
    df = pd.read_sas(RAW / file, format='xport', encoding=None)
    df = df.replace(5.397605346934028e-79, 0.0)
    return df[cols].copy()


def write(df:pd.DataFrame, name:str):
    df.to_csv(OUT/name, index=False, encoding='utf-8-sig')


def main():
    broad=pd.read_csv(OUT/'nhanes_stage1_clean_with_missing.csv')
    exam=pd.read_csv(OUT/'nhanes_examined_clean_with_missing.csv')
    predictors=pd.read_csv(OUT/'missingness_report.csv')['variable'].tolist()
    demo=sas('DEMO_J.XPT',['SEQN','RIDSTATR'])
    bpx=sas('BPX_J.XPT',['SEQN','PEASCCT1']+[f'BPXSY{i}' for i in range(1,5)]+[f'BPXDI{i}' for i in range(1,5)])
    bmx=sas('BMX_J.XPT',['SEQN','BMDSTATS','BMIWAIST'])
    assert len(exam)==5098 and exam.SEQN.is_unique
    assert all(exam.RIDSTATR.eq(2))

    full=broad.merge(demo,on='SEQN',how='left',validate='one_to_one')
    measures=['BMXBMI','BMXWAIST','BP_SYS_MEAN','BP_DIA_MEAN']
    by_exam=(full.groupby('RIDSTATR',dropna=False)[measures]
                .agg(lambda s: int(s.isna().sum())).reset_index())
    write(by_exam,'examination_missingness_audit.csv')

    d=exam[['SEQN']].merge(bpx,on='SEQN',how='left',validate='one_to_one')
    sys_cols=[f'BPXSY{i}' for i in range(1,5)]
    dia_cols=[f'BPXDI{i}' for i in range(1,5)]
    count_sys=d[sys_cols].notna().sum(axis=1)
    count_dia=d[dia_cols].notna().sum(axis=1)
    bp_counts=pd.DataFrame({'number_of_readings':list(range(5)),
                            'systolic_participants':[(count_sys==i).sum() for i in range(5)],
                            'diastolic_participants':[(count_dia==i).sum() for i in range(5)]})
    write(bp_counts,'bp_readings_audit.csv')
    fourth_rows=[]
    for label, cols in [('Systolic',sys_cols),('Diastolic',dia_cols)]:
        avail=d[cols[3]].notna()
        n_three=d[cols[:3]].notna().sum(axis=1)
        fourth_rows.append({'measurement':label,'fourth_reading_available':int(avail.sum()),
            'fourth_with_missing_first_three':int((avail & (n_three<3)).sum()),
            'fourth_with_all_first_three':int((avail & (n_three==3)).sum())})
    write(pd.DataFrame(fourth_rows),'bp_fourth_reading_audit.csv')

    missing_bp=exam.loc[exam.BP_SYS_MEAN.isna() & exam.BP_DIA_MEAN.isna(),['SEQN']]
    reason=missing_bp.merge(bpx[['SEQN','PEASCCT1']],on='SEQN',how='left',validate='one_to_one')
    dcc={1:'Safety exclusion',2:'Participant refusal',3:'Time constraints',4:'Other reason',-1:'No reason recorded'}
    reason_tab=reason.PEASCCT1.fillna(-1).value_counts().sort_index().rename_axis('PEASCCT1_code').reset_index(name='participants')
    reason_tab['reason']=reason_tab.PEASCCT1_code.map(dcc).fillna('Unspecified')
    write(reason_tab,'bp_missing_reasons_report.csv')

    wc=exam.loc[exam.BMXWAIST.isna(),['SEQN']].merge(bmx,on='SEQN',how='left',validate='one_to_one')
    stat_map={1:'Complete for age',2:'Only height and weight',3:'Other partial body exam',4:'No body-measures data'}
    wcstat=wc.BMDSTATS.fillna(-1).value_counts().sort_index().rename_axis('BMDSTATS_code').reset_index(name='participants')
    wcstat['description']=wcstat.BMDSTATS_code.map(stat_map).fillna('Unspecified')
    write(wcstat,'waist_body_status_audit.csv')
    comment=wc.BMIWAIST.fillna(-1).value_counts().sort_index().rename_axis('BMIWAIST_code').reset_index(name='participants')
    comment['description']=comment.BMIWAIST_code.map({1:'Could not obtain',-1:'No comment recorded'}).fillna('Unspecified')
    write(comment,'waist_comment_audit.csv')

    # XPT artifact occurrences in files as delivered, before their correction.
    zeros=[]
    for filename, cols in [('DEMO_J.XPT',['WTMEC2YR','INDFMPIR']),
                           ('BPX_J.XPT',[f'BPXDI{i}' for i in range(1,5)])]:
        unpatched=pd.read_sas(RAW/filename,format='xport',encoding=None)
        eligible_ids=set(broad.SEQN.astype(int))
        matched=unpatched.loc[unpatched.SEQN.isin(eligible_ids)]
        for col in cols:
            n=int((matched[col] == 5.397605346934028e-79).sum())
            if n: zeros.append({'raw_file':filename,'variable':col,'true_zeros_decoded_as_small_positive':n,'scope':'5393 eligible participants'})
    write(pd.DataFrame(zeros),'zero_artifact_source_audit.csv')

    qa={
        'full_n':len(broad),'exam_n':len(exam),
        'full_unique_seqn':broad.SEQN.nunique(),'exam_unique_seqn':exam.SEQN.nunique(),
        'full_yes':int(broad.diabetes_reported.sum()),
        'exam_yes':int(exam.diabetes_reported.sum()),
        'full_missing_any':int(broad[predictors].isna().any(axis=1).sum()),
        'exam_missing_any':int(exam[predictors].isna().any(axis=1).sum()),
        'exam_complete':int(exam[predictors].notna().all(axis=1).sum()),
        'exam_positive_weights':int((exam.WTMEC2YR>0).sum()),
        'missing_bp_exam':len(missing_bp),'missing_waist_exam':len(wc),
        'bpx_fourth_systolic':int(d.BPXSY4.notna().sum()),
    }
    write(pd.DataFrame([{'metric':k,'value':v} for k,v in qa.items()]),'independent_qa_summary.csv')
    assert qa['full_n']==5393 and qa['exam_n']==5098
    assert qa['exam_missing_any']==1067 and qa['exam_complete']==4031
    assert len(missing_bp)==258 and len(wc)==319
    assert int(d.BPXSY4.notna().sum())==423
    print('Additional audit reports written:')
    for k,v in qa.items(): print(' ',k,v)

if __name__=='__main__':
    main()
