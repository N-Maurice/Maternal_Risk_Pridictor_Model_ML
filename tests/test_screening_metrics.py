"""Check operating-point policies against real cached validation predictions."""
from pathlib import Path
import sys,json,hashlib,unittest
import numpy as np
from sklearn.metrics import fbeta_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from screening_metrics import operating_points

class ScreeningThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache=np.load(ROOT/'data/processed/state/traditional_validation_predictions.npz')
        cls.y=cls.cache['y'];cls.p=cls.cache['hist_gradient_boosting']
        import pandas as pd
        scores=pd.read_csv(ROOT/'reports/tables/stage5_validation_models.csv')
        cls.threshold=float(scores.loc[scores.model.eq('hist_gradient_boosting'),'threshold'].iloc[0])

    def test_recall_targets_and_complete_confusion_counts(self):
        table=operating_points(self.y,self.p,self.threshold)
        for row in table.itertuples():
            self.assertEqual(row.detected_preterm+row.missed_preterm,int(self.y.sum()))
            self.assertEqual(row.detected_preterm+row.missed_preterm+row.false_alerts+row.true_negatives,len(self.y))
            self.assertAlmostEqual(row.alert_fraction,(row.detected_preterm+row.false_alerts)/len(self.y))
        for goal in [.70,.80]:
            row=table.loc[table.policy.eq(f'Exploratory recall at least {goal:.0%}')].iloc[0]
            self.assertGreaterEqual(row.recall,goal)
            higher=np.unique(self.p[self.p>row.threshold])
            if len(higher):self.assertLess(np.mean(self.p[self.y==1]>=higher[0]),goal)

    def test_f2_improvement_and_tied_score_handling(self):
        table=operating_points(self.y,self.p,self.threshold)
        self.assertGreaterEqual(table.iloc[1].f2,table.iloc[0].f2)
        # Actual majority-baseline predictions provide a real tied-score case.
        tied=operating_points(self.y,self.cache['dummy'],.5)
        self.assertEqual(tied.iloc[1].recall,1.)
        self.assertEqual(tied.iloc[1].alert_fraction,1.)

    def test_protected_predictions_and_test_guard_are_unchanged(self):
        manifest=json.loads((ROOT/'reports/tables/core_scope_provenance.json').read_text())
        for name,expected in manifest['protected_evidence_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)
        damaged=self.p.copy();damaged[0]=np.nan
        with self.assertRaises(ValueError):operating_points(self.y,damaged,self.threshold)

if __name__=='__main__':unittest.main(verbosity=2)
