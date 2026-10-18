"""Execute either or both maternal notebooks, each in a fresh review kernel."""
from pathlib import Path
import argparse,sys,time,hashlib
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[1]
NAMES=['01_data_preparation_and_eda.ipynb','02_models_and_evaluation.ipynb','03_lean_feature_evaluation.ipynb']
parser=argparse.ArgumentParser()
parser.add_argument('--notebook',choices=['1','2','3'],help='Default: run the two core notebooks; 3 selects the separate feature challenger')
parser.add_argument('--input-root',type=Path)
parser.add_argument('--output-root',type=Path)
parser.add_argument('--output-dir',type=Path,help='Destination for executed notebooks; defaults to notebooks/')
parser.add_argument('--train-feature-study',action='store_true',help='Train notebook 3; executed training copy goes under experiments/')
args=parser.parse_args()
if args.train_feature_study and args.notebook!='3':parser.error('--train-feature-study requires --notebook 3')
source_root=args.input_root.resolve() if args.input_root else ROOT
state=source_root/'data/processed/state/test_evaluation_guard.json'
before=hashlib.sha256(state.read_bytes()).hexdigest()
selected=[NAMES[int(args.notebook)-1]] if args.notebook else NAMES[:2]
for name in selected:
    notebook=nbformat.read(ROOT/'notebooks'/name,as_version=4)
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        cell.outputs=[];cell.execution_count=None
        if args.train_feature_study:cell.source=cell.source.replace('RUN_STUDY = False','RUN_STUDY = True')
        if args.input_root:
            cell.source=cell.source.replace('INPUT_ROOT_OVERRIDE = None',f'INPUT_ROOT_OVERRIDE = {str(args.input_root.resolve())!r}')
        if args.output_root:
            cell.source=cell.source.replace('OUTPUT_ROOT_OVERRIDE = None',f'OUTPUT_ROOT_OVERRIDE = {str(args.output_root.resolve())!r}')
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,timeout=600,km=km,resources={'metadata':{'path':str(ROOT)}})
    start=time.perf_counter()
    try:client.execute()
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    assert hashlib.sha256(state.read_bytes()).hexdigest()==before
    count=sum(c.cell_type=='code' for c in notebook.cells)
    notebook.metadata['review_execution']={'mode':'feature-study-training' if args.train_feature_study else 'review','engine':'nbclient, fresh IPython kernel',
        'code_cells_executed':count,'failures':[],'elapsed_seconds':time.perf_counter()-start,
        'test_guard_unchanged':True,'hosted_kaggle_execution':False,'independent_kernel':True,
        'read_only_input_simulation':bool(args.input_root)}
    nbformat.validate(notebook)
    default_output=ROOT/'experiments/feature_challenger/executed' if args.train_feature_study else ROOT/'notebooks'
    output=(args.output_dir or default_output)/name
    output.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,output)
    print(f'PASS: {name}: {count} code cells, {len(notebook.cells)} total cells',flush=True)
