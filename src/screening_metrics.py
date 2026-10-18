"""Validation-only operating-point comparisons; no fitted models or test access."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,confusion_matrix,precision_score,recall_score,fbeta_score

def operating_points(y,p,recorded_threshold,recall_targets=(.70,.80)):
    y,p=np.asarray(y),np.asarray(p)
    if y.ndim!=1 or p.shape!=y.shape or set(np.unique(y))!={0,1}:
        raise ValueError('Require aligned binary labels with both classes and one score per row.')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Scores must be finite probabilities between zero and one.')
    precision,recall,thresholds=precision_recall_curve(y,p)
    f2=5*precision[:-1]*recall[:-1]/np.maximum(4*precision[:-1]+recall[:-1],1e-15)
    best=np.flatnonzero(np.isclose(f2,f2.max(),rtol=0,atol=1e-14))[-1]
    policies=[('Recorded validation-F1 threshold',float(recorded_threshold)),
              ('Exploratory maximum validation F2',float(thresholds[best]))]
    for goal in recall_targets:
        if not 0<goal<=1:raise ValueError('Recall target must be in (0,1].')
        eligible=np.flatnonzero(recall[:-1]>=goal)
        # Highest threshold satisfying the empirical recall goal minimizes alert volume.
        policies.append((f'Exploratory recall at least {goal:.0%}',float(thresholds[eligible[-1]])))
    rows=[]
    for policy,threshold in policies:
        pred=p>=threshold
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        rows.append(dict(policy=policy,threshold=threshold,precision=precision_score(y,pred,zero_division=0),
            recall=recall_score(y,pred),f2=fbeta_score(y,pred,beta=2,zero_division=0),
            specificity=tn/(tn+fp),negative_predictive_value=tn/max(1,tn+fn),
            detected_preterm=int(tp),missed_preterm=int(fn),false_alerts=int(fp),true_negatives=int(tn),
            alert_fraction=float(pred.mean()),alerts_per_1000=float(pred.mean()*1000),
            missed_preterm_per_1000=float(fn/len(y)*1000),false_alerts_per_1000=float(fp/len(y)*1000)))
    return pd.DataFrame(rows)
