"""Apply approved reporting-flag corrections without rereading all selected raw columns."""
from pathlib import Path
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ingestion import verify_layout, build_dictionary


def finalize(root):
    root=Path(root)
    audit=root/'data/processed/stage1_audit'
    tables=root/'reports/tables'
    raw_source=root/'Data/Nat2024PublicUS.c20250512.r20250708.txt'
    fingerprint=json.loads((audit/'raw_stream_audit.json').read_text())
    assert raw_source.stat().st_size==fingerprint['raw_bytes'] and raw_source.stat().st_mtime_ns==fingerprint['raw_mtime_ns']
    layout=pd.read_csv(root/'Data/natality2024_variable_layout.csv',keep_default_na=False)
    extra=dict(field='F_RF_INF_DRG',start_1based=329,end_1based=329,length=1,
               pandas_colspec_start=328,pandas_colspec_end=329,description='Reporting Flag for Fertility Enhancing Drugs',role='REPORTING_FLAG',note='Added from official guide PDF page 27; input layout preserved')
    derived=pd.concat([layout,pd.DataFrame([extra])],ignore_index=True).sort_values('start_1based')
    derived_path=tables/'stage1_corrected_variable_layout.csv'
    derived.to_csv(derived_path,index=False)
    selected,evidence,control=verify_layout(derived_path,root/'Data/UserGuide2024.pdf',audit)
    evidence.to_csv(tables/'stage1_guide_position_verification_final.csv',index=False)
    dictionary,rules=build_dictionary(selected,evidence)
    dictionary.loc[dictionary.field.eq('RF_FEDRG'),'reporting_flag']='F_RF_INF_DRG'
    dictionary['flag_zero_policy']='Associated verified flag=0 overrides populated field to NaN; flags audit only'
    dictionary.to_csv(tables/'stage1_data_dictionary.csv',index=False)
    cache=root/'data/processed/natality2024_selected_full.parquet'
    source=pq.ParquetFile(cache)
    existing=audit/'final_summary.json'
    if source.metadata.num_columns==len(selected) and existing.exists():
        summary=json.loads(existing.read_text())
        if summary.get('status')=='FINALIZED' and summary.get('parquet_bytes')==cache.stat().st_size:
            print('Approved reporting-flag correction already finalized; returning saved final missingness.')
            print(json.dumps(summary,indent=2))
            return pd.read_csv(tables/'stage1_missingness_final.csv')
    raw=root/'Data/Nat2024PublicUS.c20250512.r20250708.txt'
    old=json.loads((audit/'raw_stream_audit.json').read_text())
    assert raw.stat().st_size==old['raw_bytes'] and raw.stat().st_mtime_ns==old['raw_mtime_ns']
    flags=pd.read_fwf(raw,colspecs=[(328,329)],names=['F_RF_INF_DRG'],dtype=str,keep_default_na=False,chunksize=25000,encoding='ascii')
    temporary=cache.with_suffix('.corrected.partial')
    writer=None; rows=0; counts={f:0 for f in selected.field}; added=0
    try:
        for batch,flag in zip(source.iter_batches(batch_size=25000),flags,strict=True):
            frame=batch.to_pandas().reset_index(drop=True)
            assert len(frame)==len(flag)
            assert flag.iloc[:,0].isin(['0','1']).all()
            frame['F_RF_INF_DRG']=pd.to_numeric(flag.iloc[:,0]).to_numpy(dtype=float)
            for r in dictionary.itertuples():
                if r.reporting_flag:
                    before=frame[r.field].isna().sum()
                    frame.loc[frame[r.reporting_flag].eq(0),r.field]=float('nan') if r.kind=='numeric' else pd.NA
                    added+=int(frame[r.field].isna().sum()-before)
            frame=frame[selected.field.tolist()]
            for f in frame: counts[f]+=int(frame[f].isna().sum())
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(temporary,table.schema,compression='zstd')
            writer.write_table(table); rows+=len(frame)
    finally:
        if writer: writer.close()
    assert rows==old['raw_rows']
    source.close()
    temporary.replace(cache)
    missing=pd.DataFrame([dict(field=f,role=selected.set_index('field').loc[f,'role'],raw_rows=rows,missing_count=counts[f],missing_pct=100*counts[f]/rows) for f in selected.field])
    missing.to_csv(tables/'stage1_missingness_final.csv',index=False)
    pq.ParquetFile(cache).read_row_group(0).slice(0,5).to_pandas().to_csv(audit/'parsed_sample_final.csv',index=False)
    report=dict(raw_rows=rows,parsed_rows=rows,selected_columns=len(selected),reporting_flag_override=True,additional_missing_cells=added,rows_dropped=0,raw_sha256=old['sha256'],parquet_bytes=cache.stat().st_size,status='FINALIZED')
    (audit/'final_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (tables/'stage1_final_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return missing

if __name__=='__main__':
    finalize(Path.cwd())
