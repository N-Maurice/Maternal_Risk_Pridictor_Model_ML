"""Audit the two maternal analysis notebooks and preserved evidence without test predictions."""
from pathlib import Path
import json,hashlib,subprocess,zipfile
import nbformat,pandas as pd
from IPython.core.inputtransformer2 import TransformerManager
ROOT=Path(__file__).resolve().parents[1]
names=['01_data_preparation_and_eda.ipynb','02_models_and_evaluation.ipynb','03_lean_feature_evaluation.ipynb']
notebooks=[];code=[];per_notebook=[]
for name in names:
    n=nbformat.read(ROOT/name,as_version=4);nbformat.validate(n)
    current=[c for c in n.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in current)
    assert not n.metadata.review_execution.failures
    assert n.metadata.review_execution.independent_kernel
    assert not any(o.output_type=='error' for c in current for o in c.outputs)
    for c in current:compile(TransformerManager().transform_cell(c.source),name,'exec')
    charts=sum('image/png' in o.get('data',{}) for c in current for o in c.outputs)
    per_notebook.append(dict(notebook=name,cells=len(n.cells),executed_code_cells=len(current),embedded_charts=charts))
    notebooks.append(n);code.extend(current)
assert set(p.name for p in ROOT.glob('*.ipynb'))==set(names)
assert not (ROOT/'analysis.ipynb').exists()
audit=json.loads((ROOT/'reports/tables/stage1_audit_summary.json').read_text())
# The raw file is kept only inside the official zip; check the archived member's size.
raw=zipfile.ZipFile(ROOT/'Data/Nat2024us.zip').getinfo('Nat2024PublicUS.c20250512.r20250708.txt')
assert raw.file_size==audit['raw_bytes']
assert (ROOT/'Data/UserGuide2024.pdf').is_file()
state=ROOT/'data/processed/state/test_evaluation_guard.json'
manifest=json.loads((ROOT/'reports/revision/tables/study_manifest.json').read_text())
assert hashlib.sha256(state.read_bytes()).hexdigest()==manifest['original_test_guard_sha256']
registry=pd.read_csv(ROOT/'reports/tables/core/model_validation_comparison.csv')
assert len(registry)==8 and registry.f2.notna().all()
assert set(registry.cache_key)=={'dummy','plain_logistic','logistic','random_forest','hist_gradient_boosting','svm','dl_sequential','dl_functional'}
provenance=json.loads((ROOT/'reports/tables/core_scope_provenance.json').read_text())
for name,digest in provenance['protected_evidence_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
report=(ROOT/'reports/REPORT_condensed_source.md').read_text()
# Body words: abstract to conclusion, excluding tables, figure/table captions, references and appendices.
body=report.split('# Abstract',1)[1].split('# References',1)[0]
words=sum(len(line.split()) for line in body.splitlines() if not line.startswith(('|','![','**TABLE','```','<w:',':::')))
assert 3500<=words<=5000,words
for heading in ['Introduction','Literature Review','Methodology','Results','Discussion','Conclusion']:
    assert f'. {heading}' in report,heading
assert '{{' not in report and 'redeveloped' not in report.lower()
# Final report: the Google Docs PDF export; figures are counted in its Markdown source.
pdf=ROOT/'reports/REPORT.pdf'
assert pdf.stat().st_size>10000 and pdf.read_bytes()[:5]==b'%PDF-'
images=sum(line.startswith('![') for line in report.splitlines())
ignored_paths=['Data/'+raw.filename,'data/processed/modeling_sample.parquet','models/best_traditional.joblib','experiments/check.csv']
ignored=subprocess.run(['git','check-ignore',*ignored_paths],cwd=ROOT,capture_output=True,text=True,check=True)
assert len(ignored.stdout.splitlines())==len(ignored_paths)
tracked=subprocess.run(['git','ls-files','Data/','data/','models/','*.zip','*.parquet'],cwd=ROOT,capture_output=True,text=True,check=True)
assert not [p for p in tracked.stdout.split() if not p.endswith('.gitkeep')]
results=dict(notebooks=per_notebook,notebook_cells=sum(len(n.cells) for n in notebooks),executed_code_cells=len(code),
 execution_errors=0,raw_rows=audit['raw_rows'],raw_sha256=audit['sha256'],user_guide_preserved=True,
 main_comparison_entries=len(registry),report_body_words=words,report_figures=images,
 original_test_guard_unchanged=True,new_test_predictions=0,raw_data_tracked=False,
 report_authorship_review_pending=True,remote_hosted_account_execution=False)
(ROOT/'reports/tables/project_audit.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
