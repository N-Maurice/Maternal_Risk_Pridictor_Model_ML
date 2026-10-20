"""Write scikit-learn evaluation metrics into TensorBoard next to the Keras training logs.

Keras only reports its own streaming metrics. I compute the evaluation metrics with
scikit-learn (the same functions used in the notebooks and report tables) and write
them as TensorBoard scalars, PR curves, images, text and HParams so that classical and
neural models can be compared in one dashboard.
"""
from pathlib import Path
import io,json
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score,average_precision_score,precision_score,recall_score,
    f1_score,fbeta_score,accuracy_score,balanced_accuracy_score,matthews_corrcoef,brier_score_loss,
    log_loss,confusion_matrix,classification_report,precision_recall_curve,roc_curve)
from sklearn.calibration import calibration_curve

SWEEP=np.round(np.arange(.01,1.0,.01),2)


def best_f1_threshold(y,p):
    """Validation threshold rule used throughout the project: maximum F1."""
    precision,recall,thresholds=precision_recall_curve(y,p)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(thresholds[np.argmax(f1)])


def sklearn_scores(y,p,threshold):
    y=np.asarray(y).astype(int);p=np.asarray(p,dtype=float).ravel();pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),
        precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),
        f1=f1_score(y,pred,zero_division=0),f2=fbeta_score(y,pred,beta=2,zero_division=0),
        accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),
        specificity=tn/max(1,tn+fp),mcc=matthews_corrcoef(y,pred),brier=brier_score_loss(y,p),
        log_loss=log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),threshold=float(threshold),
        alerts_per_1000=float(pred.mean()*1000),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def _image(fig):
    import matplotlib.pyplot as plt
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110,bbox_inches='tight');plt.close(fig)
    import tensorflow as tf
    return tf.image.decode_png(buffer.getvalue(),channels=3)[None]


def _figures(y,p,threshold,title):
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    y=np.asarray(y).astype(int);p=np.asarray(p,dtype=float).ravel();pred=(p>=threshold).astype(int)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    cm=confusion_matrix(y,pred,labels=[0,1])
    ConfusionMatrixDisplay(cm,display_labels=['term','preterm']).plot(ax=axes[0],colorbar=False,values_format='d')
    ConfusionMatrixDisplay(cm/cm.sum(axis=1,keepdims=True),display_labels=['term','preterm']).plot(ax=axes[1],colorbar=False,values_format='.3f')
    axes[0].set_title(f'{title}: counts');axes[1].set_title('Row-normalized')
    confusion=_image(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fpr,tpr,_=roc_curve(y,p);axes[0].plot(fpr,tpr,label=f'AUC={roc_auc_score(y,p):.3f}');axes[0].plot([0,1],[0,1],'--',color='grey')
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC');axes[0].legend()
    precision,recall,_=precision_recall_curve(y,p);axes[1].plot(recall,precision,label=f'AP={average_precision_score(y,p):.3f}')
    axes[1].axhline(y.mean(),ls='--',color='grey',label='Prevalence');axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall');axes[1].legend()
    observed,predicted=calibration_curve(y,p,n_bins=10,strategy='quantile')
    axes[2].plot(predicted,observed,'o-');axes[2].plot([0,1],[0,1],'--',color='grey')
    axes[2].set(xlabel='Mean predicted probability',ylabel='Observed preterm fraction',title='Calibration')
    fig.suptitle(title)
    return confusion,_image(fig)


def _hparam_value(value):
    if isinstance(value,(bool,int,float,str)) and not (isinstance(value,float) and not np.isfinite(value)):return value
    if isinstance(value,np.generic):return value.item()
    return json.dumps(value,default=str) if not isinstance(value,str) else value


def log_evaluation(logdir,y,p,threshold,description,params=None,step=0,split='validation'):
    """Write one model's scikit-learn evaluation to TensorBoard.

    Scalars: ``sklearn_<split>/<metric>``; PR curve; confusion-matrix and ROC/PR/calibration
    images; a classification report; a threshold sweep (step = threshold in hundredths);
    and HParams so the HParams dashboard can rank models by the same metrics.
    """
    import tensorflow as tf
    from tensorboard.plugins.pr_curve import summary as pr_summary
    from tensorboard.plugins.hparams import api as hp
    y=np.asarray(y).astype(int);p=np.asarray(p,dtype=float).ravel()
    scores=sklearn_scores(y,p,threshold);title=Path(logdir).name
    writer=tf.summary.create_file_writer(str(logdir))
    with writer.as_default():
        if params is not None:
            hp.hparams({str(k):_hparam_value(v) for k,v in params.items()},trial_id=title)
        for key,value in scores.items():tf.summary.scalar(f'sklearn_{split}/{key}',float(value),step=step)
        pb=pr_summary.pb(f'sklearn_{split}/pr_curve',labels=y.astype(bool),predictions=np.clip(p,0,1).astype('float32'),num_thresholds=201,
                         display_name='Precision–recall',description='scikit-learn style PR curve from model probabilities')
        tf.summary.experimental.write_raw_pb(pb.SerializeToString(),step=step)
        confusion,curves=_figures(y,p,threshold,f'{title} ({split})')
        tf.summary.image(f'sklearn_{split}/confusion_matrix',confusion,step=step)
        tf.summary.image(f'sklearn_{split}/roc_pr_calibration',curves,step=step)
        report=pd.DataFrame(classification_report(y,p>=threshold,labels=[0,1],target_names=['term','preterm'],output_dict=True,zero_division=0)).T.round(4)
        tf.summary.text(f'sklearn_{split}/classification_report','```\n'+report.to_string()+'\n```',step=step)
        tf.summary.text('description',description,step=step)
        for t in SWEEP:
            pred=p>=t;index=int(round(t*100))
            tf.summary.scalar('threshold_sweep/precision',precision_score(y,pred,zero_division=0),step=index)
            tf.summary.scalar('threshold_sweep/recall',recall_score(y,pred,zero_division=0),step=index)
            tf.summary.scalar('threshold_sweep/f1',f1_score(y,pred,zero_division=0),step=index)
            tf.summary.scalar('threshold_sweep/f2',fbeta_score(y,pred,beta=2,zero_division=0),step=index)
            tf.summary.scalar('threshold_sweep/alerts_per_1000',float(pred.mean()*1000),step=index)
    writer.flush();writer.close()
    return scores


def log_cv_results(logdir,results,description):
    """Hyperparameter search: one step per candidate, in the order scikit-learn sampled them."""
    import tensorflow as tf
    results=pd.DataFrame(results)
    writer=tf.summary.create_file_writer(str(logdir))
    with writer.as_default():
        for step,row in enumerate(results.to_dict('records')):
            for key in ['mean_test_score','mean_train_score','std_test_score','mean_fit_time']:
                if key in row and np.isfinite(row[key]):tf.summary.scalar(f'cv_search/{key}',float(row[key]),step=step)
        columns=[c for c in ['params','mean_train_score','mean_test_score','std_test_score','rank_test_score'] if c in results]
        tf.summary.text('cv_search/candidates','```\n'+results[columns].round(4).to_string()+'\n```',step=0)
        tf.summary.text('description',description,step=0)
    writer.flush();writer.close()


def log_learning_curve(logdir,sizes,train_scores,cv_scores,metric,description):
    """Learning curve with step = number of training records."""
    import tensorflow as tf
    writer=tf.summary.create_file_writer(str(logdir))
    with writer.as_default():
        for size,a,b in zip(sizes,np.atleast_2d(train_scores),np.atleast_2d(cv_scores)):
            tf.summary.scalar(f'learning_curve/train_{metric}',float(np.mean(a)),step=int(size))
            tf.summary.scalar(f'learning_curve/cv_{metric}',float(np.mean(b)),step=int(size))
        tf.summary.text('description',description,step=0)
    writer.flush();writer.close()


def log_recorded_history(logdir,history,description):
    """Recorded Keras epoch history from a completed fit (step = epoch)."""
    import tensorflow as tf
    history=pd.DataFrame(history).drop(columns=['epoch'],errors='ignore')
    writer=tf.summary.create_file_writer(str(logdir))
    with writer.as_default():
        tf.summary.text('description',description,step=0)
        for epoch,row in enumerate(history.to_dict('records'),start=1):
            for key,value in row.items():
                if np.isfinite(value):tf.summary.scalar(f'recorded_epoch/{key}',float(value),step=epoch)
    writer.flush();writer.close()


def sklearn_epoch_callback(logdir,inputs,y,batch_size=1024):
    """Keras callback: after every epoch, predict the validation set and log scikit-learn metrics."""
    import tensorflow as tf
    y=np.asarray(y).astype(int)

    class SklearnValidationMetrics(tf.keras.callbacks.Callback):
        def on_train_begin(self,logs=None):
            self.writer=tf.summary.create_file_writer(str(logdir))
            self.dataset=tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)
        def on_epoch_end(self,epoch,logs=None):
            p=self.model.predict(self.dataset,verbose=0).ravel()
            scores=sklearn_scores(y,p,best_f1_threshold(y,p))
            with self.writer.as_default():
                for key in ['roc_auc','average_precision','precision','recall','f1','f2','brier','log_loss','threshold']:
                    tf.summary.scalar(f'sklearn_epoch/{key}',float(scores[key]),step=epoch+1)
            self.writer.flush()
        def on_train_end(self,logs=None):
            self.writer.close()
    return SklearnValidationMetrics()
