"""Real-data checks for the isolated feature comparison."""
from pathlib import Path
import json,sys,unittest
import numpy as np
import pandas as pd
import nbformat,joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from project import CORE_NUM,CORE_CAT
from readable_features import readable_features

class FeatureChallengerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=ROOT/'reports/tables/feature_challenger'
        cls.plan=json.loads((cls.directory/'study_plan.json').read_text())
        cls.data=pd.read_parquet(ROOT/'data/processed/modeling_sample.parquet',columns=CORE_NUM+CORE_CAT+['row_id','target'])
        cls.ids=np.load(ROOT/'data/processed/state/split_indices.npz')

    def test_challenger_schema_and_executed_notebook(self):
        schema=self.plan['features']['challenger_16']
        self.assertEqual(len(schema),16)
        self.assertEqual(len(set(schema)),16)
        self.assertFalse(set(schema)&{'preterm_birth','no_previous_pregnancy_indicator',
            'recorded_history_risk_count','live_birth_order_category','total_birth_order_category',
            'prepregnancy_weight_pounds','previous_preterm_birth_indicator'})
        n=nbformat.read(ROOT/'notebooks/03_lean_feature_evaluation.ipynb',as_version=4)
        nbformat.validate(n)
        self.assertTrue(n.metadata.review_execution.independent_kernel)
        for c in n.cells:
            if c.cell_type=='code':
                compile(c.source,'feature-notebook','exec')
                self.assertIsNotNone(c.execution_count)
                self.assertFalse(any(o.output_type=='error' for o in c.outputs))

    def test_lean_imputation_uses_training_only_and_models_reload(self):
        schema=self.plan['features']['challenger_16']
        train=self.data.iloc[self.ids['train']]
        x=readable_features(train,engineered=False)[schema]
        for family in ['hist_gradient_boosting','logistic']:
            model=joblib.load(ROOT/'models/feature_challenger'/f'{family}__challenger_16.joblib')
            prep=model.named_steps['preprocessing']
            numeric=prep.transformers_[0][2]
            observed=prep.named_transformers_['numeric'].named_steps['impute'].statistics_
            np.testing.assert_allclose(observed,np.nanmedian(x[numeric].to_numpy(),axis=0))
            before=observed.copy()
            self.assertTrue(np.isfinite(model.predict_proba(x.head(5))).all())
            np.testing.assert_array_equal(before,prep.named_transformers_['numeric'].named_steps['impute'].statistics_)

    def test_validation_alignment_and_original_reference_reproduction(self):
        cached=np.load(ROOT/'models/feature_challenger/validation_predictions.npz')
        val=self.data.iloc[self.ids['validation']]
        np.testing.assert_array_equal(cached['row_id'],val.row_id)
        np.testing.assert_array_equal(cached['y'],val.target)
        original=np.load(ROOT/'data/processed/state/traditional_validation_predictions.npz')
        for family in ['hist_gradient_boosting','logistic']:
            np.testing.assert_allclose(cached[f'{family}__current_32'],original[family],rtol=0,atol=1e-12)
        self.assertEqual(self.plan['primary_comparison'],'hist_gradient_boosting')
        self.assertTrue(self.plan['feature_proposal_informed_by_previous_validation'])

if __name__=='__main__':unittest.main(verbosity=2)
