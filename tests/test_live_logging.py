"""Verify live logging and the scikit-learn → TensorBoard link using small new runs."""
from pathlib import Path
import sys,tempfile,json,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from live_logging import RunLog


def tags(folder):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(folder),size_guidance={'tensors':0}).Reload()
    return events,events.Tags()['tensors']


class LiveLoggingTests(unittest.TestCase):
    def test_review_logs_saved_prediction_metrics_but_no_training(self):
        rng=np.random.default_rng(0);y=rng.integers(0,2,400);p=np.clip(y*.3+rng.random(400)*.7,0,1)
        with tempfile.TemporaryDirectory() as folder:
            run=RunLog(folder,'review','review')
            self.assertEqual(run.path.parent.name,'review')
            with self.assertRaises(RuntimeError):run.keras_callbacks('cached')
            scores=run.evaluation('cached model',y,p,.5,source='saved_completed_validation_predictions',params={'C':1.0,'class_weight':None})
            run.finish()
            self.assertFalse((run.path/'neural').exists())
            events,names=tags(run.path/'sklearn_metrics/validation/cached_model')
            for tag in ['sklearn_validation/roc_auc','sklearn_validation/average_precision','sklearn_validation/recall',
                        'sklearn_validation/pr_curve/pr_curves','sklearn_validation/confusion_matrix','threshold_sweep/recall']:
                self.assertIn(tag,names)
            from sklearn.metrics import roc_auc_score
            from tensorboard.util import tensor_util
            logged=float(tensor_util.make_ndarray(events.Tensors('sklearn_validation/roc_auc')[0].tensor_proto))
            self.assertAlmostEqual(logged,roc_auc_score(y,p),places=6)
            self.assertAlmostEqual(scores['roc_auc'],roc_auc_score(y,p))
            self.assertIn('saved_completed_validation_predictions',(run.path/'events.jsonl').read_text())
            self.assertEqual(json.loads((run.path/'manifest.json').read_text())['status'],'completed')

    def test_failure_is_recorded_and_run_ids_are_unique(self):
        with tempfile.TemporaryDirectory() as folder:
            run=RunLog(folder,'trial','training');other=RunLog(folder,'trial','training')
            self.assertNotEqual(run.path,other.path)
            with self.assertRaises(ValueError):
                with run.operation('fit'):raise ValueError('test failure')
            self.assertIn('operation_failed',(run.path/'events.jsonl').read_text())
            self.assertEqual(json.loads((run.path/'manifest.json').read_text())['status'],'started')

    def test_real_fit_writes_keras_and_sklearn_epochs(self):
        import tensorflow as tf
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            run=RunLog(folder,'logging_smoke_test','training')
            x=np.arange(16,dtype=np.float32).reshape(-1,1)/16
            y=(x[:,0]>.5).astype(np.float32)
            options=tf.data.Options();options.threading.private_threadpool_size=1
            ds=tf.data.Dataset.from_tensor_slices((x,y)).batch(4).with_options(options)
            model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(1,activation='sigmoid')])
            model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
            model.fit(ds,validation_data=ds,epochs=2,callbacks=run.keras_callbacks('tiny_model',validation=(x,y)),verbose=0)
            run.finish()
            self.assertEqual(len(pd.read_csv(run.path/'neural/tiny_model/epochs.csv')),2)
            events,names=tags(run.path/'neural/tiny_model/tensorboard/train')
            self.assertEqual(len(events.Tensors('epoch_loss')),2)
            events,names=tags(run.path/'neural/tiny_model/tensorboard/sklearn_validation')
            self.assertEqual(len(events.Tensors('sklearn_epoch/roc_auc')),2)
            self.assertEqual((run.path/'events.jsonl').read_text().count('epoch_completed'),2)

    def test_cv_and_learning_curve_dashboards(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            run=RunLog(folder,'search','training')
            run.cv_results('lr',pd.DataFrame({'params':[{'C':1},{'C':10}],'mean_test_score':[.3,.31],'mean_train_score':[.32,.33],'std_test_score':[.01,.01],'rank_test_score':[2,1]}),'new')
            run.learning_curve('lr',[100,400],[[.8,.8],[.75,.76]],[[.6,.62],[.68,.69]],'new')
            events,_=tags(run.path/'sklearn_search/lr');self.assertEqual(len(events.Tensors('cv_search/mean_test_score')),2)
            events,_=tags(run.path/'learning_curves/lr');self.assertEqual([e.step for e in events.Tensors('learning_curve/cv_roc_auc')],[100,400])

if __name__=='__main__':unittest.main()
