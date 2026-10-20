"""Stage 6: Sequential and Functional Keras models with tf.data pipelines."""
import os,time,json,gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from project import *
from data_stages import pipeline
from traditional import stratified_subset


def seed_tensorflow(cfg,offset=0):
    tf.keras.utils.set_random_seed(cfg.seed+offset)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    tf.config.experimental.enable_op_determinism()


class EmbeddingPreprocessor:
    """Train-only numeric normalization and category indices; zero is unknown."""
    def fit(self,x):
        self.numeric,self.categorical=feature_names()
        self.num=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])
        self.num.fit(x[self.numeric])
        self.encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,dtype=np.int32)
        self.encoder.fit(x[self.categorical])
        self.vocab_sizes=[len(c)+1 for c in self.encoder.categories_]
        self.numeric_dim=self.num.transform(x[self.numeric].iloc[:1]).shape[1]
        return self
    def transform(self,x):
        encoded=self.encoder.transform(x[self.categorical])+1
        result={'numeric':self.num.transform(x[self.numeric]).astype('float32')}
        result.update({'c_'+f.lower():encoded[:,i:i+1].astype('int32') for i,f in enumerate(self.categorical)})
        return result


def dataset(inputs,y=None,batch_size=512,training=False,seed=42):
    ds=tf.data.Dataset.from_tensor_slices(inputs if y is None else (inputs,np.asarray(y,dtype=np.float32)))
    n=len(next(iter(inputs.values()))) if isinstance(inputs,dict) else len(inputs)
    if training:ds=ds.shuffle(min(n,20000),seed=seed,reshuffle_each_iteration=True)
    options=tf.data.Options();options.deterministic=True;options.threading.private_threadpool_size=4;options.threading.max_intra_op_parallelism=1
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).with_options(options)


def architecture(kind,preprocessor,params,input_dim=None):
    k=tf.keras
    if kind=='sequential':
        model=k.Sequential([k.layers.Input(shape=(input_dim,),name='encoded_features')],name='sequential_mlp')
        for i in range(params['depth']):
            model.add(k.layers.Dense(params['units'],activation='relu',kernel_regularizer=k.regularizers.l2(params['l2'])))
            model.add(k.layers.BatchNormalization());model.add(k.layers.Dropout(params['dropout']))
        model.add(k.layers.Dense(1,activation='sigmoid'))
    else:
        inputs={'numeric':k.Input(shape=(preprocessor.numeric_dim,),name='numeric')};encoded=[inputs['numeric']]
        for f,vocab in zip(preprocessor.categorical,preprocessor.vocab_sizes):
            name='c_'+f.lower();inp=k.Input(shape=(1,),dtype='int32',name=name);inputs[name]=inp
            emb=k.layers.Embedding(vocab,min(8,max(2,int(np.ceil(np.sqrt(vocab))))),name='embedding_'+f.lower())(inp)
            encoded.append(k.layers.Flatten()(emb))
        z=k.layers.Concatenate()(encoded)
        for i in range(params['depth']):
            z=k.layers.Dense(params['units'],activation='relu',kernel_regularizer=k.regularizers.l2(params['l2']))(z)
            z=k.layers.BatchNormalization()(z);z=k.layers.Dropout(params['dropout'])(z)
        model=k.Model(inputs,k.layers.Dense(1,activation='sigmoid')(z),name='functional_embeddings')
    optimizer=(k.optimizers.Adam if params['optimizer']=='adam' else k.optimizers.RMSprop)(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=[k.metrics.AUC(name='roc_auc'),k.metrics.AUC(name='pr_auc',curve='PR')])
    return model


def callbacks(path, run_log=None, name=None):
    live = run_log.keras_callbacks(name or Path(path).stem) if run_log is not None else []
    return live + [tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=4,restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc',mode='max',patience=2,factor=.5,min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(path,monitor='val_pr_auc',mode='max',save_best_only=True)]


def predict_model(model,preprocessor,frame):
    x=preprocessor.transform(features(frame))
    if not isinstance(x,dict):x=np.asarray(x,dtype=np.float32)
    return model.predict(dataset(x,batch_size=1024),verbose=0).ravel()


def stage6(cfg):
    if (cfg.state/'test_evaluation_guard.json').exists():
        raise RuntimeError('Final test has been opened; training/tuning is frozen. Use saved results or start a separate explicitly documented study.')
    from live_logging import RunLog
    run_log=RunLog(cfg.root,'stage6',mode='training',seed=cfg.seed)
    run_log.fingerprint([cfg.sample,cfg.state/'split_indices.npz'])
    seed_tensorflow(cfg)
    train,val=load_dev(cfg);xt,xv=features(train),features(val);yt=train.target.to_numpy();yv=val.target.to_numpy()
    sub=stratified_subset(yt,cfg.tuning_size,cfg.seed)
    weights={0:len(yt)/(2*(yt==0).sum()),1:len(yt)/(2*(yt==1).sum())}
    rng=np.random.default_rng(cfg.seed)
    space=dict(depth=[1,2,3],units=[32,64,96],dropout=[.15,.3],learning_rate=[.0005,.001],batch_size=[512,1024],optimizer=['adam','rmsprop'],l2=[.0001,.001])
    trials=[{k:(lambda value:value.item() if isinstance(value,np.generic) else value)(rng.choice(v)) for k,v in space.items()} for _ in range(3)]
    # Record realized settings before any fits. Search is bounded random search, not exhaustive.
    write_json(cfg.tables/'stage6_search_space.json',dict(space=space,trials_per_api=3,realized_trials=trials,max_epochs_per_trial=16,final_max_epochs=24,tuning_training_rows=len(sub),final_training_rows=len(train),class_weights=weights,selection='external sklearn validation average_precision after best val_pr_auc checkpoint',seed=cfg.seed))
    tune_rows=[];final_rows=[];val_preds={}
    for kind in ['sequential','functional']:
        if kind=='sequential':prep=pipeline();prep.fit(xt)
        else:prep=EmbeddingPreprocessor().fit(xt)
        joblib.dump(prep,cfg.models/f'dl_{kind}_preprocessing.joblib')
        a,b=prep.transform(xt),prep.transform(xv)
        if kind=='sequential':a=np.asarray(a,dtype='float32');b=np.asarray(b,dtype='float32')
        subinputs={k:v[sub] for k,v in a.items()} if isinstance(a,dict) else a[sub]
        kind_trials=[];search_start=time.perf_counter()
        for trial,params in enumerate(trials):
            tf.keras.backend.clear_session();seed_tensorflow(cfg,trial)
            model=architecture(kind,prep,params,input_dim=None if isinstance(a,dict) else a.shape[1])
            start=time.perf_counter();path=cfg.models/f'trial_{kind}_{trial}.keras'
            history=model.fit(dataset(subinputs,yt[sub],params['batch_size'],True,cfg.seed+trial),validation_data=dataset(b,yv,params['batch_size']),epochs=16,class_weight=weights,callbacks=callbacks(path,run_log),shuffle=False,verbose=0)
            model=tf.keras.models.load_model(path)
            p=model.predict(dataset(b,batch_size=1024),verbose=0).ravel()
            row=dict(api=kind,trial=trial,phase='tuning',training_rows=len(sub),epochs_run=len(history.history['loss']),training_seconds=time.perf_counter()-start,parameters=json.dumps(params),**metrics(yv,p,choose_threshold(yv,p)))
            run_log.result(f'{kind}_trial_{trial}',row)
            kind_trials.append(row);tune_rows.append(row)
            pd.DataFrame(history.history).to_csv(cfg.tables/f'stage6_history_{kind}_trial{trial}.csv',index=False)
            print('DL TRIAL',row,flush=True)
            del model;gc.collect()
        winning=max(kind_trials,key=lambda r:r['average_precision']);params=json.loads(winning['parameters'])
        tf.keras.backend.clear_session();seed_tensorflow(cfg)
        model=architecture(kind,prep,params,input_dim=None if isinstance(a,dict) else a.shape[1]);path=cfg.models/f'dl_{kind}.keras'
        start=time.perf_counter()
        history=model.fit(dataset(a,yt,params['batch_size'],True,cfg.seed),validation_data=dataset(b,yv,params['batch_size']),epochs=24,class_weight=weights,callbacks=callbacks(path,run_log),shuffle=False,verbose=0)
        model=tf.keras.models.load_model(path)
        p=model.predict(dataset(b,batch_size=1024),verbose=0).ravel();threshold=choose_threshold(yv,p)
        row=dict(model='dl_'+kind,api=kind,training_rows=len(train),epochs_run=len(history.history['loss']),best_epoch=int(np.argmax(history.history['val_pr_auc'])+1),training_seconds=time.perf_counter()-search_start,refit_seconds=time.perf_counter()-start,best_parameters=json.dumps(params),**metrics(yv,p,threshold))
        if row['roc_auc']>.85:raise RuntimeError('Core DL AUC exceeds leakage trigger; investigate.')
        run_log.result('dl_'+kind,row)
        final_rows.append(row);val_preds['dl_'+kind]=p
        hist=pd.DataFrame(history.history);hist['epoch']=np.arange(1,len(hist)+1);hist.to_csv(cfg.tables/f'stage6_history_{kind}.csv',index=False)
        fig,axes=plt.subplots(1,3,figsize=(13,3.5))
        for ax,key in zip(axes,['loss','roc_auc','pr_auc']):ax.plot(hist.epoch,hist[key],label='train');ax.plot(hist.epoch,hist['val_'+key],label='validation');ax.set_title(kind+' '+key);ax.set_xlabel('Epoch');ax.legend()
        savefig(cfg,f'stage6_history_{kind}')
        with (cfg.tables/f'stage6_architecture_{kind}.md').open('w') as f:model.summary(print_fn=lambda line:f.write(line+'\n'))
        print('DL FINAL',row,flush=True)
        del model,a,b;gc.collect()
    tuning=pd.DataFrame(tune_rows);tuning.to_csv(cfg.tables/'stage6_tuning.csv',index=False)
    final=pd.DataFrame(final_rows).sort_values('average_precision',ascending=False);final.to_csv(cfg.tables/'stage6_validation_models.csv',index=False)
    best=final.iloc[0];write_json(cfg.state/'deep_selection.json',dict(deep=best.model,threshold=float(best.threshold),selection_metric='validation average_precision',test_opened=False))
    np.savez_compressed(cfg.state/'deep_validation_predictions.npz',y=yv,**val_preds)
    write_json(cfg.tables/'stage6_fit_provenance.json',dict(encoder_fit_split='train',fit_rows=len(train),validation_fit_rows=0,test_fit_rows=0,tf_data_operations=['from_tensor_slices','shuffle (train only)','batch','prefetch'],functional_oov_index=0))
    stage_done(cfg,6,final.to_dict('records'))
    run_log.finish()
    return final
