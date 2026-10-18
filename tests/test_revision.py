"""Meaningful boundaries for revised calibration, serialization and portability."""
from pathlib import Path
import unittest,sys,json,tempfile,hashlib,ast,base64,io,zipfile
import numpy as np
import pandas as pd
import joblib,nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from project import Config,load_dev,stored_or_new_partitions
from readable_features import readable_features,NAMES,readable_names
from notebook_runtime import configure_inputs,verify_final_test_scope

class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=Config(ROOT).setup();cls.train,cls.val=load_dev(cls.cfg)

    def test_calibration_and_early_stop_are_disjoint_training_subsets(self):
        ids=np.load(self.cfg.models/'revision/training_partition_indices.npz')
        fit,cal,deep,stop=[set(ids[k]) for k in ['fit','calibration','deep_fit','deep_early_stop']]
        self.assertFalse(fit&cal);self.assertFalse(deep&stop)
        self.assertEqual(deep|stop,fit);self.assertEqual(fit|cal,set(range(len(self.train))))

    def test_native_categories_use_only_fitting_rows_and_handle_unknown(self):
        model=joblib.load(self.cfg.models/'revision/native_hist_boosting.joblib')
        ids=np.load(self.cfg.models/'revision/training_partition_indices.npz')['fit']
        x=readable_features(self.train.iloc[ids]);transform=model.named_steps['categories']
        for col,values in transform.categories_.items():self.assertEqual(values,sorted(x[col].dropna().unique().tolist()))
        row=x.head(1).copy();col='maternal_education';row[col]='new_unseen_category'
        self.assertTrue(transform.transform(row)[col].isna().all())
        self.assertTrue(np.isfinite(model.predict_proba(row)).all())

    def test_readable_contract_is_reduced_and_target_free(self):
        result=readable_features(self.train.head(20))
        self.assertEqual(result.shape[1],32)
        self.assertFalse(set(result.columns)&{'preterm_birth','source_record_id','obstetric_gestation_weeks','delivery_payment_category','pregnancy_weight_gain_pounds'})
        self.assertTrue(all(name in NAMES.values() for name in result))

    def test_test_guard_unchanged_and_revision_has_no_test_claim(self):
        manifest=json.loads((ROOT/'reports/revision/tables/study_manifest.json').read_text())
        current=hashlib.sha256((self.cfg.state/'test_evaluation_guard.json').read_bytes()).hexdigest()
        self.assertEqual(current,manifest['original_test_guard_sha256']);self.assertEqual(manifest['original_test_predictions'],0)
        table=pd.read_csv(ROOT/'reports/revision/tables/validation_metrics.csv')
        self.assertTrue(table.scope.eq('development_validation_only').all())

    def test_portable_inputs_are_linked_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);source=base/'input';work=base/'working';source.mkdir();work.mkdir()
            # Real training rows: no synthetic modeling dataset is introduced.
            path=source/'modeling_sample.parquet';self.train.head(3).to_parquet(path,index=False)
            checksum=hashlib.sha256(path.read_bytes()).hexdigest()
            configure_inputs(work,source)
            self.assertTrue((work/'data/processed/modeling_sample.parquet').is_symlink())
            self.assertEqual(checksum,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_existing_split_manifest_is_respected_with_different_runtime_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=Config(Path(temp),seed=987).setup()
            frame=self.train.head(10)
            path=cfg.state/'split_indices.npz'
            expected=[np.array([0,2,4,6,7,8,9]),np.array([1,5]),np.array([3])]
            np.savez(path,train=expected[0],validation=expected[1],test=expected[2])
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            for actual,wanted in zip(stored_or_new_partitions(frame,cfg),expected):np.testing.assert_array_equal(actual,wanted)
            self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_copied_original_cohort_cannot_be_relabelled_as_a_new_test(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=Config(Path(temp)).setup()
            (cfg.data/'input_provenance.json').write_text(json.dumps({'original_test_already_evaluated':True}))
            with self.assertRaisesRegex(RuntimeError,'previously evaluated'):
                verify_final_test_scope(cfg)

    def test_two_notebooks_match_authorized_core_scope(self):
        manifest=json.loads((ROOT/'reports/tables/notebook_split_manifest.json').read_text())
        found={}
        sources=[]
        for name in manifest['new_notebooks']:
            nb=nbformat.read(ROOT/name,as_version=4);nbformat.validate(nb)
            for i,cell in enumerate(nb.cells):
                if cell.cell_type=='code':
                    compile(cell.source,'notebook-cell','exec')
                    sources.append(cell.source)
                    found[(name,i)]=hashlib.sha256(cell.source.encode()).hexdigest()
        for row in manifest['cells']:
            self.assertEqual(found[(row['notebook'],row['cell_index'])],row['sha256'])
        self.assertEqual(len(found),len(manifest['cells']))
        source='\n'.join(sources)
        for required in ['LogisticRegression','RandomForestClassifier','HistGradientBoostingClassifier',
                         'SVC','tf.keras.Sequential','tf.keras.Model','from_tensor_slices','operating_points']:
            self.assertIn(required,source)
        for removed in ['SMOTENC','LGBMClassifier','ExtraTreesClassifier','AveragedProbabilityEnsemble','DEVELOPMENT_TABLES']:
            self.assertNotIn(removed,source)
        self.assertFalse((ROOT/'analysis.ipynb').exists())
    def test_historical_registry_records_validation_only_and_f2(self):
        from experiment_log import build_registry
        registry=build_registry(self.cfg,tensorboard=False)
        self.assertEqual(len(registry),19)
        self.assertTrue(registry.evaluation_scope.eq('development_validation').all())
        self.assertTrue(registry.f2.between(0,1).all())
        self.assertTrue(registry.record_type.eq('saved_experiment_result').all())

if __name__=='__main__':unittest.main(verbosity=2)
