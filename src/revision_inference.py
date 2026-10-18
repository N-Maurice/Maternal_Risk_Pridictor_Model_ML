"""Inference for the development-only revision; accepts readable or CDC raw columns."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from readable_features import readable_features,readable_names,NAMES

class RevisedPredictor:
    def __init__(self,model_dir,model_name=None):
        self.directory=Path(model_dir)
        self.manifest=json.loads((self.directory/'selected_candidate.json').read_text())
        self.name=model_name or self.manifest['selected_candidate']
        self._models={}
    def _inputs(self,frame):
        num,cat=readable_names();expected=num+cat
        if set(expected).issubset(frame.columns):return frame[expected].copy()
        return readable_features(frame)
    def _component(self,name,x):
        calibrated=name.endswith('_sigmoid');base=name.removesuffix('_sigmoid')
        if base.startswith(('native_hist','extra_trees')):
            if base not in self._models:self._models[base]=joblib.load(self.directory/f'{base}.joblib')
            p=self._models[base].predict_proba(x)[:,1]
        else:
            import tensorflow as tf
            from deep_learning import dataset
            if base not in self._models:
                self._models[base]=(tf.keras.models.load_model(self.directory/f'{base}.keras'),joblib.load(self.directory/f'{base}_preprocessing.joblib'))
            model,prep=self._models[base]
            if hasattr(prep,'categorical') and not set(prep.categorical).issubset(x.columns):
                # Legacy internal embedding names stay traceable; caller uses readable aliases.
                reverse={v:k for k,v in NAMES.items()};values=prep.transform(x.rename(columns=reverse))
            else:values=prep.transform(x)
            if not isinstance(values,dict):values=np.asarray(values,dtype='float32')
            p=model.predict(dataset(values,batch_size=1024),verbose=0).ravel()
        if calibrated:p=joblib.load(self.directory/f'{base}_sigmoid.joblib').predict(p)
        return p
    def predict_proba(self,frame):
        x=self._inputs(frame)
        if self.name=='equal_weight_tree_neural_ensemble':
            p=np.mean([self._component(n,x) for n in self.manifest['ensemble_members']],axis=0)
        else:p=self._component(self.name,x)
        return np.column_stack([1-p,p])
    def predict(self,frame):
        if self.name!=self.manifest['selected_candidate']:raise ValueError('Load the selected candidate to use its recorded threshold.')
        return (self.predict_proba(frame)[:,1]>=self.manifest['threshold']).astype('int8')
