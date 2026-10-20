"""Execution logs: live Keras training events plus scikit-learn evaluation dashboards in TensorBoard."""
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata
import hashlib,json,platform,re,sys,time,uuid
from contextlib import contextmanager

class RunLog:
    def __init__(self, root, name, mode, seed=42):
        self.root=Path(root)
        self.mode=mode
        self.path=self.root/'runs'/('live' if mode=='training' else 'review')/f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{self.safe(name)}_{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True)
        packages={}
        for package in ['numpy','pandas','scikit-learn','tensorflow-cpu','keras','tensorboard','pyarrow']:
            try: packages[package]=metadata.version(package)
            except metadata.PackageNotFoundError: pass
        manifest=dict(name=name,mode=mode,seed=seed,created_utc=datetime.now(timezone.utc).isoformat(),
            python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,
            provenance='New execution; live callbacks only capture training performed during this run.',
            status='started')
        self.manifest=manifest
        self._save_manifest()
        self.record('run_started',mode=mode)

    @staticmethod
    def safe(name): return re.sub(r'[^A-Za-z0-9_.-]+','_',str(name))

    def _save_manifest(self):
        (self.path/'manifest.json').write_text(json.dumps(self.manifest,indent=2,default=str)+'\n')

    def record(self,event,**values):
        row=dict(utc=datetime.now(timezone.utc).isoformat(),event=event,**values)
        with (self.path/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')

    def fingerprint(self, paths):
        checks={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths if Path(p).is_file()}
        self.manifest.setdefault('input_sha256',{}).update(checks);self._save_manifest()

    def keras_callbacks(self,name,validation=None):
        """TensorBoard + CSVLogger + lifecycle events; with ``validation=(inputs,y)`` also
        scikit-learn validation metrics after every epoch, in the same TensorBoard run."""
        if self.mode!='training':raise RuntimeError('Live training callbacks require a training run.')
        import tensorflow as tf
        path=self.path/'neural'/self.safe(name);path.mkdir(parents=True,exist_ok=False)
        self.record('neural_fit_configured',model=name,log_directory=str(path))
        run=self
        class Lifecycle(tf.keras.callbacks.Callback):
            def on_train_begin(self,logs=None):run.record('neural_fit_started',model=name)
            def on_epoch_end(self,epoch,logs=None):run.record('epoch_completed',model=name,epoch=epoch+1,metrics=logs or {})
            def on_train_end(self,logs=None):run.record('neural_fit_finished',model=name)
        result=[tf.keras.callbacks.TensorBoard(log_dir=str(path/'tensorboard'),update_freq='epoch',profile_batch=0),
                tf.keras.callbacks.CSVLogger(str(path/'epochs.csv')),Lifecycle()]
        if validation is not None:
            from tensorboard_metrics import sklearn_epoch_callback
            result.insert(1,sklearn_epoch_callback(path/'tensorboard'/'sklearn_validation',*validation))
        return result

    def result(self,name,values,source='new_development_validation_only'):
        self.record('model_result',model=name,source=source,metrics=values)

    def evaluation(self,name,y,p,threshold,source,params=None,split='validation',group=None):
        """scikit-learn metrics for one model's probabilities, written to TensorBoard.

        ``source`` states where the probabilities came from. Saved predictions from the
        completed study are labelled as such; they are evaluations, not training events."""
        from tensorboard_metrics import log_evaluation
        folder=self.path/'sklearn_metrics'/(group or split)/self.safe(name)
        description=(f'**{name}** — {split} split. Probabilities source: `{source}`. '
                     f'Run mode: `{self.mode}`. Metrics computed with scikit-learn {self.manifest["packages"].get("scikit-learn","")}.')
        scores=log_evaluation(folder,y,p,threshold,description,params=params,split=split)
        self.record('sklearn_metrics_logged',model=name,split=split,source=source,log_directory=str(folder),
                    metrics={k:v for k,v in scores.items() if k in ('roc_auc','average_precision','precision','recall','f1','brier')})
        return scores

    def cv_results(self,name,results,source):
        from tensorboard_metrics import log_cv_results
        log_cv_results(self.path/'sklearn_search'/self.safe(name),results,f'{name} randomized search; source: `{source}`.')
        self.record('cv_results_logged',model=name,source=source,candidates=len(results))

    def learning_curve(self,name,sizes,train_scores,cv_scores,source,metric='roc_auc'):
        from tensorboard_metrics import log_learning_curve
        log_learning_curve(self.path/'learning_curves'/self.safe(name),sizes,train_scores,cv_scores,metric,
                           f'{name} learning curve (training-fold CV); source: `{source}`.')
        self.record('learning_curve_logged',model=name,source=source)

    def recorded_history(self,name,history,source):
        """Epoch history saved by the completed study's training run; shown with its source."""
        from tensorboard_metrics import log_recorded_history
        log_recorded_history(self.path/'recorded_histories'/self.safe(name),history,
                             f'Epoch history saved by the completed training run (`{source}`); written to TensorBoard during this {self.mode} execution.')
        self.record('recorded_history_logged',model=name,source=source)

    @contextmanager
    def operation(self,name,**settings):
        self.record('operation_started',name=name,settings=settings);started=time.perf_counter()
        try:yield
        except BaseException as error:
            self.record('operation_failed',name=name,error=repr(error));raise
        else:self.record('operation_finished',name=name,seconds=time.perf_counter()-started)

    def table(self,name,frame):
        target=self.path/'tables'/f'{self.safe(name)}.csv';target.parent.mkdir(exist_ok=True)
        frame.to_csv(target,index=False)
        self.record('table_saved',name=name,rows=len(frame),path=str(target))

    def finish(self):
        self.manifest.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
        self._save_manifest();self.record('run_completed')
