"""Stages 2–4: stratified sampling, development-only EDA and train-only preprocessing."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from project import *


def stage2(cfg):
    if (cfg.state/'test_evaluation_guard.json').exists():
        print('Final evaluation is frozen; returning saved stage2 results without resampling or fitting.')
        return pd.read_csv(cfg.tables/'stage2_class_balance.csv')
    assert (cfg.data/'stage1_audit/final_summary.json').is_file()
    counts={1:0,2:0,3:0}; total=0
    source=pq.ParquetFile(cfg.cache)
    for b in source.iter_batches(batch_size=25000,columns=['OEGest_R3']):
        y=b.column(0).to_numpy(zero_copy_only=False); total+=len(y)
        counts[1]+=int((y==1).sum()); counts[2]+=int((y==2).sum()); counts[3]+=int(np.isnan(y).sum())
    valid=counts[1]+counts[2]; n=min(cfg.sample_size,valid)
    quotas={1:round(n*counts[1]/valid)}; quotas[2]=n-quotas[1]
    rng=np.random.default_rng(cfg.seed); reservoirs={k:(np.array([],dtype=float),np.array([],dtype=np.int64)) for k in [1,2]}
    offset=0
    for b in source.iter_batches(batch_size=25000,columns=['OEGest_R3']):
        y=b.column(0).to_numpy(zero_copy_only=False); ids=offset+np.arange(len(y)); offset+=len(y)
        priorities=rng.random(len(y))
        for k in [1,2]:
            mask=y==k; oldkeys,oldids=reservoirs[k]
            keys=np.concatenate([oldkeys,priorities[mask]]); rowids=np.concatenate([oldids,ids[mask]])
            keep=np.argpartition(keys,min(quotas[k],len(keys))-1)[:quotas[k]]
            reservoirs[k]=(keys[keep],rowids[keep])
    chosen=np.zeros(total,dtype=bool)
    for k in [1,2]: chosen[reservoirs[k][1]]=True
    frames=[]; offset=0
    for b in source.iter_batches(batch_size=25000):
        mask=chosen[offset:offset+len(b)]; frame=b.to_pandas()
        frame['row_id']=offset+np.arange(len(b)); offset+=len(b)
        if mask.any(): frames.append(frame.loc[mask])
    sample=pd.concat(frames,ignore_index=True).sort_values('row_id').reset_index(drop=True)
    sample['target']=sample.OEGest_R3.eq(1).astype('int8')
    assert len(sample)==n and sample.row_id.is_unique and sample.OEGest_R3.notna().all()
    sample.to_parquet(cfg.sample,index=False,compression='zstd')
    balance=pd.DataFrame([dict(population='full_known_target',rows=valid,preterm=counts[1],term=counts[2],preterm_ratio=counts[1]/valid),dict(population='stratified_sample',rows=n,preterm=int(sample.target.sum()),term=int((1-sample.target).sum()),preterm_ratio=float(sample.target.mean()))])
    balance.to_csv(cfg.tables/'stage2_class_balance.csv',index=False)
    report=dict(raw_rows=total,target_code3_dropped=counts[3],known_target_rows=valid,sample_rows=n,seed=cfg.seed,sampling='Exact per-class random-priority reservoir, uniform without replacement',ratio_absolute_difference=abs(sample.target.mean()-counts[1]/valid),justification='300k bounds local tuning/embedding memory while retaining a substantial minority class; CV and SVM use documented training-only subsamples. Full-population class counts are computed before sampling.')
    assert report['ratio_absolute_difference']<=1/n
    write_json(cfg.tables/'stage2_sampling.json',report)
    print(balance.to_string(index=False)); print(json.dumps(report,indent=2))
    stage_done(cfg,2,report)
    return balance


def pipeline(kind='linear',engineered=True,time_features=False):
    nums,cats=feature_names(engineered,time_features)
    num=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])
    cat=Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='__MISSING__')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32))])
    return ColumnTransformer([('numeric',num,nums),('categorical',cat,cats)],remainder='drop',verbose_feature_names_out=True)


def leakage_table(cfg):
    x=pd.read_csv(cfg.tables/'stage1_corrected_variable_layout.csv',keep_default_na=False)
    core=set(CORE_NUM+CORE_CAT)
    times=set(TIME_NUM+TIME_CAT)
    rows=[]
    for r in x.itertuples():
        if r.field in core: status,reason='KEEP','History, prepregnancy characteristic, demographic, or prenatal plurality; recorded values may be imperfect proxies.'
        elif r.field in times: status,reason='ABLATION','Accumulates during pregnancy or has unrecorded onset/availability; excluded from core.'
        elif r.field=='ILP_R': status,reason='EXCLUDE','Numeric interval uses infant birth date (guide technical note p55); only no-prior-pregnancy structural flag can be retained.'
        elif r.field in ['PRECARE','PRECARE5']: status,reason='EXCLUDE','Month of care is calculated using obstetric gestation (guide technical note p55–56).'
        elif r.field=='PAY_REC': status,reason='EXCLUDE','Delivery payment source; reporting/subgroups only, prenatal value not established.'
        elif r.field=='SEX': status,reason='EXCLUDE','Recorded at birth; prenatal ascertainment not established.'
        elif r.field=='BMI_R': status,reason='EXCLUDE','Redundant with BMI; engineered BMI categories use explicit standard thresholds.'
        elif r.role=='TARGET': status,reason='EXCLUDE','Outcome/gestation source; never a predictor.'
        elif r.role=='REPORTING_FLAG': status,reason='EXCLUDE','Reporting audit/cleaning only, including outcome reporting flags.'
        elif r.role=='EXCLUDE_leakage': status,reason='EXCLUDE',r.note or 'Delivery/outcome/date variable.'
        else: status,reason='EXCLUDE','Not approved by prenatal availability audit; do not infer suitability from an unspecified role.'
        rows.append(dict(field=r.field,input_role=r.role,decision=status,reason=reason))
    out=pd.DataFrame(rows);out.to_csv(cfg.tables/'stage3_leakage_audit.csv',index=False)
    return out


def stage3(cfg):
    if (cfg.state/'test_evaluation_guard.json').exists():
        print('Final evaluation is frozen; returning saved stage3 results without resampling or fitting.')
        return pd.read_csv(cfg.tables/'stage3_preterm_rates.csv')
    sample=pd.read_parquet(cfg.sample)
    train,val,test=stored_or_new_partitions(sample,cfg)
    # EDA and sanity check explicitly exclude prospective held-out rows.
    dev=sample.iloc[np.concatenate([train,val])].copy()
    x=features(dev); y=dev.target
    groups={'age_band':pd.cut(dev.MAGER,[-np.inf,19,24,29,34,39,np.inf]).astype(str),
            'bmi_category':pd.cut(dev.BMI,[-np.inf,18.5,25,30,35,40,np.inf],right=False).astype(str),
            'prior_preterm':dev.RF_PPTERM.fillna('missing'), 'plurality':dev.DPLURAL.fillna(-1).astype(str),
            'education':dev.MEDUC.fillna(-1).astype(str),'insurance':dev.PAY_REC.fillna(-1).astype(str)}
    tables=[]
    for name,g in groups.items():
        table=pd.DataFrame({'group':g,'target':y}).groupby('group').target.agg(['size','sum','mean']).reset_index()
        table['variable']=name; table.columns=['group','rows','preterm','preterm_rate','variable'];tables.append(table)
        plt.figure(figsize=(8,4));plt.bar(table.group.astype(str),table.preterm_rate);plt.ylabel('Preterm proportion');plt.xlabel(name);plt.xticks(rotation=30,ha='right');savefig(cfg,f'stage3_rate_{name}')
    rates=pd.concat(tables,ignore_index=True);rates.to_csv(cfg.tables/'stage3_preterm_rates.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,f in zip(axes,['MAGER','BMI','PRIORLIVE']): ax.hist(dev[f].dropna(),bins=30);ax.set_title(f)
    savefig(cfg,'stage3_numeric_distributions')
    a,b=train_test_split(np.arange(len(dev)),test_size=.2,stratify=y,random_state=cfg.seed)
    sanity=Pipeline([('preprocessing',pipeline()),('model',DecisionTreeClassifier(max_depth=3,min_samples_leaf=100,random_state=cfg.seed))])
    sanity.fit(x.iloc[a],y.iloc[a]);score=roc_auc_score(y.iloc[b],sanity.predict_proba(x.iloc[b])[:,1])
    importances=pd.DataFrame({'feature':sanity.named_steps['preprocessing'].get_feature_names_out(),'importance':sanity.named_steps['model'].feature_importances_}).sort_values('importance',ascending=False)
    importances.to_csv(cfg.tables/'stage3_shallow_tree_importance.csv',index=False)
    audit=leakage_table(cfg)
    report=dict(development_rows=len(dev),reserved_test_rows=len(test),sanity_roc_auc=score,depth=3,auc_investigation_trigger=.85,implausibly_high_importance_flags=importances.loc[importances.importance>.8,'feature'].tolist(),investigation='No AUC trigger' if score<=.85 else 'STOP: investigate before continuing',importance_review='Plurality dominance is plausible given known multiple-pregnancy risk; quantify with singleton-only ablation.',short_interpregnancy_interval='Not safely derivable: ILP_R uses current birth date. No target gestation subtraction is permitted.',limitations=['Row-level data lack maternal/pregnancy identifiers; repeated pregnancies or siblings may cross splits. Singleton sensitivity and cautious generalization required.','No geography or clinical biomarkers; all availability claims remain retrospective proxies.'])
    write_json(cfg.tables/'stage3_sanity.json',report)
    print(audit.decision.value_counts());print(json.dumps(report,indent=2));print(importances.head(10).to_string(index=False))
    if score>.85: raise RuntimeError('AUC leakage trigger; stop before modeling.')
    stage_done(cfg,3,report)
    return rates


def stage4(cfg):
    if (cfg.state/'test_evaluation_guard.json').exists():
        print('Final evaluation is frozen; returning saved stage4 results without resampling or fitting.')
        return pd.read_csv(cfg.tables/'stage4_split_manifest.csv')
    sample=pd.read_parquet(cfg.sample)
    train,val,test=stored_or_new_partitions(sample,cfg)
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    assert len(train)+len(val)+len(test)==len(sample)
    split_path=cfg.state/'split_indices.npz'
    if not split_path.exists():np.savez(split_path,train=train,validation=val,test=test)
    rows=[]
    for name,idx in [('train',train),('validation',val),('test_locked',test)]:
        # Test labels used for stratification only; no test metrics or fitting.
        rows.append(dict(split=name,rows=len(idx),row_id_sha256=hashlib.sha256(sample.row_id.iloc[idx].to_numpy().tobytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(cfg.tables/'stage4_split_manifest.csv',index=False)
    x=features(sample.iloc[train]); transform=pipeline();transform.fit(x)
    joblib.dump(transform,cfg.models/'core_preprocessing.joblib')
    pd.DataFrame({'output_feature':transform.get_feature_names_out()}).to_csv(cfg.tables/'stage4_transformed_features.csv',index=False)
    nums,cats=feature_names()
    write_json(cfg.tables/'stage4_fit_provenance.json',dict(fit_split='train',fit_rows=len(train),validation_fit_rows=0,test_fit_rows=0,stateless_engineering=True,numeric=nums,categorical=cats,transformed_dimension=len(transform.get_feature_names_out()),missing_indicators='Numeric SimpleImputer(add_indicator=True); categorical missing category',experiments=['Baseline versus engineered','With versus without time-dependent features','Singleton-only']))
    print(pd.DataFrame(rows).to_string(index=False));print(transform);print('Output dimensions:',len(transform.get_feature_names_out()))
    stage_done(cfg,4,rows)
    return pd.DataFrame(rows)
