"""Real-data checks for leakage boundaries, fitted transforms and frozen evaluation."""
from pathlib import Path
import sys,json,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from project import Config,features,load_dev


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=Config(Path(__file__).resolve().parents[1]).setup()
        cls.train,cls.val=load_dev(cls.cfg)

    def test_split_ids_are_disjoint_and_exhaustive(self):
        ids=np.load(self.cfg.state/'split_indices.npz')
        sets=[set(ids[k]) for k in ['train','validation','test']]
        self.assertEqual([len(s) for s in sets],[210000,45000,45000])
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(set.union(*sets)),300000)

    def test_predictor_boundary_and_stateless_engineering(self):
        raw=self.train.head(100).copy(deep=True);before=raw.copy(deep=True)
        result=features(raw)
        pd.testing.assert_frame_equal(raw,before)
        excluded={'ILP_R','PRECARE','PRECARE5','PAY_REC','SEX','OEGest_R3','OEGest_Comb','GESTREC3','COMBGEST','WTGAIN','PREVIS','CIG_2','CIG_3'}
        self.assertFalse(excluded&set(result.columns))
        self.assertFalse(any(f.startswith(('F_','DOB_')) for f in result))
        pd.testing.assert_frame_equal(result,features(raw))

    def test_stored_preprocessing_medians_are_training_only(self):
        prep=joblib.load(self.cfg.models/'core_preprocessing.joblib')
        nums=prep.transformers_[0][2]
        observed=prep.named_transformers_['numeric'].named_steps['impute'].statistics_
        expected=np.nanmedian(features(self.train)[nums].to_numpy(),axis=0)
        np.testing.assert_allclose(observed,expected)
        state=observed.copy();prep.transform(features(self.val.head(100)))
        np.testing.assert_array_equal(prep.named_transformers_['numeric'].named_steps['impute'].statistics_,state)

    def test_test_guard_reuses_metrics_without_model_loading(self):
        from evaluation import stage8
        guard=json.loads((self.cfg.state/'test_evaluation_guard.json').read_text())
        self.assertEqual(guard['status'],'complete');self.assertEqual(guard['test_predictions_per_model'],1)
        saved=pd.read_csv(self.cfg.tables/'stage8_final_test_metrics.csv')
        with patch('joblib.load',side_effect=AssertionError('No model loading allowed on completed evaluation')):
            returned=stage8(self.cfg)
        pd.testing.assert_frame_equal(returned,saved)

    def test_post_test_training_is_blocked(self):
        from traditional import stage5
        with self.assertRaisesRegex(RuntimeError,'frozen'):
            stage5(self.cfg)

    def test_required_reloads_passed(self):
        proof=pd.read_csv(self.cfg.tables/'stage9_reload_proof.csv')
        self.assertTrue(proof[proof.artifact.ne('best_deep.h5')].passed.all())

if __name__=='__main__':unittest.main(verbosity=2)
