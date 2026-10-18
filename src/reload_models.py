"""Demonstrate frozen artifact inference on TRAINING rows only (no refit/test access)."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from project import Config,features,load_dev
from deep_learning import seed_tensorflow,predict_model


def main():
    cfg=Config(Path(__file__).resolve().parents[1]).setup();seed_tensorflow(cfg)
    train,_=load_dev(cfg);rows=train.head(10)
    traditional=joblib.load(cfg.models/'best_traditional.joblib')
    prep=joblib.load(cfg.models/'best_deep_preprocessing.joblib')
    deep=tf.keras.models.load_model(cfg.models/'best_deep.keras')
    p=traditional.predict_proba(features(rows))[:,1];q=predict_model(deep,prep,rows)
    print(pd.DataFrame({'source_row_id':rows.row_id,'traditional_preterm_probability':p,'deep_preterm_probability':q}).to_string(index=False))
    assert np.isfinite(p).all() and np.isfinite(q).all()
    print('Reloaded both models and predicted on training rows without fitting.')

if __name__=='__main__':main()
