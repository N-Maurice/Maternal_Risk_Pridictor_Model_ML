"""Shared paths, reproducibility, feature policy and metrics."""
from dataclasses import dataclass
from pathlib import Path
import os,random,json,time
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR','/tmp/maternal-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,accuracy_score,brier_score_loss,precision_recall_curve)

@dataclass
class Config:
    root: Path
    seed: int=42
    sample_size: int=300000
    tuning_size: int=60000
    svm_size: int=6000
    folds: int=3
    jobs: int=2
    bootstrap_reps: int=500
    @property
    def data(self): return self.root/'data/processed'
    @property
    def sample(self): return self.data/'modeling_sample.parquet'
    @property
    def cache(self): return self.data/'natality2024_selected_full.parquet'
    @property
    def models(self): return self.root/'models'
    @property
    def tables(self): return self.root/'reports/tables'
    @property
    def figures(self): return self.root/'reports/figures'
    @property
    def state(self): return self.data/'state'
    def setup(self):
        for p in (self.data,self.models,self.tables,self.figures,self.state): p.mkdir(parents=True,exist_ok=True)
        random.seed(self.seed); np.random.seed(self.seed)
        os.environ['PYTHONHASHSEED']=str(self.seed)
        os.environ.setdefault('OMP_NUM_THREADS','4')
        os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
        os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
        os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
        plt.rcParams.update({'figure.dpi':110,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
        return self

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def savefig(cfg,name):
    plt.tight_layout(); plt.savefig(cfg.figures/f'{name}.png',dpi=300,bbox_inches='tight'); plt.close()

def metrics(y,p,threshold=.5):
    pred=np.asarray(p)>=threshold
    return dict(roc_auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
                average_precision=average_precision_score(y,p),precision=precision_score(y,pred,zero_division=0),
                recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),
                accuracy=accuracy_score(y,pred),brier=brier_score_loss(y,p),threshold=float(threshold))

def choose_threshold(y,p):
    # Fixed rule before test access: maximize validation F1. No test information.
    precision,recall,t=precision_recall_curve(y,p)
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(t[np.argmax(f)])

CORE_NUM=['MAGER','FAGECOMB','PRIORLIVE','PRIORDEAD','PRIORTERM','CIG_0','M_Ht_In','BMI','PWgt_R','RF_CESARN']
CORE_CAT=['MRACEHISP','DMAR','MEDUC','LBO_REC','TBO_REC','RF_PDIAB','RF_PHYPE','RF_PPTERM','RF_INFTR','RF_FEDRG','RF_ARTEC','RF_CESAR','DPLURAL']
TIME_NUM=['PREVIS','CIG_1','CIG_2','CIG_3','WTGAIN']
TIME_CAT=['PREVIS_REC','WTGAIN_REC','RF_GDIAB','RF_GHYPE','RF_EHYPE','WIC','IP_GON','IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC']
ENGINEERED_NUM=['advanced_age','teen','prior_preterm','prior_pregnancy_none','risk_count','age_x_chronic_hypertension','plurality_x_prior_preterm']
ENGINEERED_CAT=['age_band','bmi_category']


def feature_names(engineered=True,time_features=False):
    return (CORE_NUM+(ENGINEERED_NUM if engineered else [])+(TIME_NUM if time_features else []),
            CORE_CAT+(ENGINEERED_CAT if engineered else [])+(TIME_CAT if time_features else []))


def features(frame,engineered=True,time_features=False):
    # Fixed, stateless transformations. No parameters are estimated here.
    x=frame.copy()
    if engineered:
        age=x.MAGER; bmi=x.BMI
        x['advanced_age']=(age>=35).astype(float).mask(age.isna())
        x['teen']=(age<20).astype(float).mask(age.isna())
        x['prior_preterm']=x.RF_PPTERM.map({'Y':1.,'N':0.})
        x['prior_pregnancy_none']=x.ILP_R.eq(888).astype(float).mask(x.ILP_R.isna())
        factors=['RF_PDIAB','RF_PHYPE','RF_PPTERM','RF_INFTR','RF_CESAR']
        risk=pd.DataFrame({f:x[f].map({'Y':1.,'N':0.}) for f in factors},index=x.index)
        # Partial missingness remains missing rather than being counted as absence.
        x['risk_count']=risk.sum(axis=1,min_count=len(factors))
        x['age_x_chronic_hypertension']=x.advanced_age*x.RF_PHYPE.map({'Y':1.,'N':0.})
        x['plurality_x_prior_preterm']=x.DPLURAL*x.prior_preterm
        x['age_band']=pd.cut(age,[-np.inf,19,24,29,34,39,np.inf],labels=['<20','20–24','25–29','30–34','35–39','40+']).astype('string')
        x['bmi_category']=pd.cut(bmi,[-np.inf,18.5,25,30,35,40,np.inf],right=False,labels=['underweight','normal','overweight','obesity_I','obesity_II','obesity_III']).astype('string')
    nums,cats=feature_names(engineered,time_features)
    x=x[nums+cats].copy()
    for f in nums: x[f]=pd.to_numeric(x[f],errors='raise').astype(float)
    for f in cats: x[f]=x[f].astype('string').fillna('__MISSING__').astype(object)
    assert not any(f.startswith('F_') or f in {'OEGest_R3','OEGest_Comb','GESTREC3','COMBGEST','ILP_R','PRECARE','PRECARE5','PAY_REC','SEX'} for f in x)
    return x


def partitions(frame,cfg):
    from sklearn.model_selection import train_test_split
    idx=np.arange(len(frame))
    dev,test=train_test_split(idx,test_size=.15,stratify=frame.target,random_state=cfg.seed)
    train,val=train_test_split(dev,test_size=15/85,stratify=frame.target.iloc[dev],random_state=cfg.seed)
    return train,val,test


def stored_or_new_partitions(frame,cfg):
    """Respect uploaded split manifests, including read-only Kaggle inputs."""
    path=cfg.state/'split_indices.npz'
    if not path.exists():return partitions(frame,cfg)
    saved=np.load(path)
    result=tuple(np.asarray(saved[name],dtype=np.int64) for name in ['train','validation','test'])
    joined=np.concatenate(result)
    if len(joined)!=len(frame) or len(np.unique(joined))!=len(frame) or joined.min()!=0 or joined.max()!=len(frame)-1:
        raise ValueError('Split manifest is not disjoint/exhaustive for this modeling sample.')
    return result


def load_dev(cfg):
    data=pd.read_parquet(cfg.sample)
    ids=np.load(cfg.state/'split_indices.npz')
    return data.iloc[ids['train']].copy(),data.iloc[ids['validation']].copy()


def stage_done(cfg,number,details):
    write_json(cfg.state/f'stage_{number:02d}.json',dict(stage=number,status='complete',details=details))
    progress=cfg.root/'PROGRESS.md'
    if progress.exists():
        text=progress.read_text().replace(f'- [ ] Stage {number}:',f'- [x] Stage {number}:')
        text+=f'\n### Executed Stage {number}\n\n```json\n'+json.dumps(details,indent=2,default=str)+'\n```\n'
        progress.write_text(text)
    print(f'Stage {number} complete:',details,flush=True)
