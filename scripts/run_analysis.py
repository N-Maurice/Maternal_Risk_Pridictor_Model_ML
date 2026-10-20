"""Execute maternal notebooks with durable execution transcripts and live training logs."""
from pathlib import Path
import argparse,sys,time,hashlib,json,traceback
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[1]
NAMES=['01_data_preparation_and_eda.ipynb','02_models_and_evaluation.ipynb','03_lean_feature_evaluation.ipynb']
parser=argparse.ArgumentParser()
parser.add_argument('--notebook',choices=['1','2','3'],help='Default: review notebooks 1 and 2')
parser.add_argument('--input-root',type=Path)
parser.add_argument('--output-root',type=Path)
parser.add_argument('--output-dir',type=Path)
parser.add_argument('--train-feature-study',action='store_true')
parser.add_argument('--train',action='store_true',help='Run new development training in notebook 2')
parser.add_argument('--deep-learning',action='store_true',help='Include live-logged neural training; requires --train')
args=parser.parse_args()
if args.train_feature_study and args.notebook!='3':parser.error('--train-feature-study requires --notebook 3')
if args.train and args.notebook!='2':parser.error('--train requires --notebook 2')
if args.deep_learning and not args.train:parser.error('--deep-learning requires --train')
source_root=args.input_root.resolve() if args.input_root else ROOT
state=source_root/'data/processed/state/test_evaluation_guard.json'
before=hashlib.sha256(state.read_bytes()).hexdigest()
selected=[NAMES[int(args.notebook)-1]] if args.notebook else NAMES[:2]
for name in selected:
    notebook=nbformat.read(ROOT/name,as_version=4)
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        cell.outputs=[];cell.execution_count=None
        if args.train_feature_study:cell.source=cell.source.replace('RUN_STUDY = False','RUN_STUDY = True')
        if args.train:cell.source=cell.source.replace('RUN_TRAINING = False','RUN_TRAINING = True')
        if args.deep_learning:cell.source=cell.source.replace('RUN_DEEP_LEARNING = False','RUN_DEEP_LEARNING = True')
        if args.input_root:cell.source=cell.source.replace('INPUT_ROOT_OVERRIDE = None',f'INPUT_ROOT_OVERRIDE = {str(args.input_root.resolve())!r}')
        if args.output_root:cell.source=cell.source.replace('OUTPUT_ROOT_OVERRIDE = None',f'OUTPUT_ROOT_OVERRIDE = {str(args.output_root.resolve())!r}')
    mode='feature-study-training' if args.train_feature_study else ('development-training' if args.train else 'review')
    output_root=args.output_root.resolve() if args.output_root else ROOT
    default_output=(output_root/'experiments/feature_challenger/executed' if args.train_feature_study else
                    output_root/'experiments/core/executed' if args.train else ROOT)
    destination=args.output_dir or default_output
    if args.train or args.train_feature_study:
        from datetime import datetime,timezone
        from uuid import uuid4
        destination=destination/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    output=destination/name;output.parent.mkdir(parents=True,exist_ok=True)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,timeout=600 if mode=='review' else 7200,km=km,resources={'metadata':{'path':str(ROOT)}})
    start=time.perf_counter();failure=None
    try:client.execute()
    except BaseException as error:
        failure=repr(error)
        raise
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
        unchanged=hashlib.sha256(state.read_bytes()).hexdigest()==before
        count=sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells)
        notebook.metadata['review_execution']={'mode':mode,'engine':'nbclient, fresh IPython kernel',
            'code_cells_executed':count,'failures':[] if failure is None else [failure],
            'elapsed_seconds':time.perf_counter()-start,'test_guard_unchanged':unchanged,
            'hosted_kaggle_execution':False,'independent_kernel':True,'read_only_input_simulation':bool(args.input_root)}
        nbformat.validate(notebook);nbformat.write(notebook,output)
        lines=[f'Mode: {mode}\nPython: {sys.executable}\nFailure: {failure}\n']
        for index,cell in enumerate(notebook.cells):
            for item in cell.get('outputs',[]):
                if item.output_type=='stream':lines.append(f'\nCell {index} {item.name}:\n'+item.text)
                elif item.output_type=='error':lines.append('\n'.join(item.traceback))
                elif 'text/plain' in item.get('data',{}):lines.append(f'\nCell {index} result:\n'+item.data['text/plain'])
        output.with_suffix('.log').write_text('\n'.join(lines))
        assert unchanged,'Original test guard changed'
    print(f'PASS: {name}: {count} executed code cells; saved {output}',flush=True)
