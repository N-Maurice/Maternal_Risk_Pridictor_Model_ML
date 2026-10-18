"""Stage 1 only: guide-backed, bounded-memory Natality ingestion and audit."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SELECTED_ROLES = {'KEEP_candidate', 'ABLATION_time_dependent', 'TARGET', 'REPORTING_FLAG'}


def stream_raw_audit(raw_path: Path, output_path: Path) -> dict:
    """Count every physical record, check padding, and hash without a full load."""
    started = time.perf_counter()
    before = raw_path.stat()
    lengths, endings = Counter(), Counter()
    digest = hashlib.sha256()
    rows = nonspace_tail = 0
    with raw_path.open('rb', buffering=8 * 1024 * 1024) as source:
        for line in source:
            digest.update(line)
            record = line.rstrip(b'\r\n')
            rows += 1
            lengths[len(record)] += 1
            endings['CRLF' if line.endswith(b'\r\n') else 'LF' if line.endswith(b'\n') else 'none'] += 1
            nonspace_tail += bool(record[1330:].strip(b' '))
    after = raw_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Raw source changed during audit.')
    result = dict(raw_path=str(raw_path), raw_bytes=after.st_size, raw_rows=rows,
                  record_length_counts=dict(lengths), line_endings=dict(endings),
                  records_with_nonspace_after_position_1330=nonspace_tail,
                  sha256=digest.hexdigest(), raw_mtime_ns=after.st_mtime_ns,
                  elapsed_seconds=time.perf_counter() - started)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if rows < 20_000:
        raise ValueError('Assignment raw-row requirement failed.')
    if min(lengths) < 1330 or nonspace_tail:
        raise ValueError('Record length/padding requires investigation before parsing.')
    return result


def verify_layout(layout_path: Path, guide_path: Path, output_dir: Path):
    """Verify every selected position/name against the local official PDF."""
    layout = pd.read_csv(layout_path, keep_default_na=False)
    selected = layout[layout.role.isin(SELECTED_ROLES)].copy().reset_index(drop=True)
    if selected.field.duplicated().any():
        raise ValueError('Duplicate selected field names.')
    assert (selected.pandas_colspec_start == selected.start_1based - 1).all()
    assert (selected.pandas_colspec_end == selected.end_1based).all()
    assert (selected.length == selected.end_1based - selected.start_1based + 1).all()
    text = subprocess.run(['pdftotext', '-layout', str(guide_path), '-'],
                          check=True, capture_output=True, text=True).stdout
    pages = text.split('\f')
    pattern = re.compile(r'^\s{0,20}(\d+)(?:-(\d+))?\s+(\d+)\s+([A-Za-z][A-Za-z_0-9]*)\b')
    entries = []
    for page_number, page in enumerate(pages, 1):
        for line in page.splitlines():
            match = pattern.match(line)
            if match:
                entries.append(dict(start=int(match[1]), end=int(match[2] or match[1]),
                                    width=int(match[3]), field=match[4], pdf_page=page_number,
                                    guide_position_line=line.strip()))
    evidence = []
    for row in selected.itertuples():
        matches = [e for e in entries if e['field'] == row.field
                   and e['start'] == row.start_1based and e['end'] == row.end_1based
                   and e['width'] == row.length]
        if len(matches) != 1:
            raise ValueError(f'Guide position verification failed for {row.field}: {matches}')
        evidence.append(dict(field=row.field, **{k: v for k, v in matches[0].items() if k != 'field'}))
    evidence = pd.DataFrame(evidence)
    evidence.to_csv(output_dir / 'guide_position_verification.csv', index=False)
    # Guide-derived metadata, not dataset measurements. Assert rather than silently guess.
    assert 'Record length: 1330' in text
    count_match = re.search(r'Record count\s+([\d,]+)\s+([\d,]+)', text)
    if not count_match:
        raise ValueError('Could not find guide control counts.')
    control = dict(guide_record_length=1330,
                   guide_us_record_count=int(count_match[1].replace(',', '')),
                   guide_territories_record_count=int(count_match[2].replace(',', '')),
                   guide_sha256=hashlib.sha256(guide_path.read_bytes()).hexdigest(),
                   layout_sha256=hashlib.sha256(layout_path.read_bytes()).hexdigest())
    (output_dir / 'guide_control.json').write_text(json.dumps(control, indent=2) + '\n')
    return selected, evidence, control


def build_dictionary(selected: pd.DataFrame, evidence: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Explicit code lists transcribed from each selected variable's guide entry.

    Numeric code strings retain guide widths for validation (e.g. 000 and 888).
    No global sentinel replacement is used. RF_* X is structural, not unknown.
    """
    specs = {}

    def numeric(field, low, high, missing=(), extras=(), step=1, flag='', note=''):
        width = int(selected.loc[selected.field.eq(field), 'length'].iloc[0])
        if step == 0.1:
            allowed = {f'{x / 10:.1f}' for x in range(round(low * 10), round(high * 10) + 1)}
        else:
            allowed = {str(x).zfill(width) for x in range(low, high + 1)}
        missing = {str(x).zfill(width) for x in missing}
        allowed |= missing | {str(x).zfill(width) for x in extras}
        specs[field] = dict(kind='numeric', allowed=sorted(allowed), missing=sorted(missing),
                            reporting_flag=flag, semantic_note=note)

    numeric('MAGER', 12, 50, note='12 groups ages 10–12; 50 groups ages 50+. Age is edited/imputed upstream.')
    numeric('MRACEHISP', 1, 7, missing=[8], note='Protected/social grouping; suitability for prediction requires Stage 3 review.')
    numeric('DMAR', 1, 2, note='US codes only. Puerto Rico 3/9 are not accepted in this US file.')
    numeric('MEDUC', 1, 8, missing=[9], flag='F_MEDUC')
    numeric('FAGECOMB', 9, 98, missing=[99])
    for field in ['PRIORLIVE', 'PRIORDEAD', 'PRIORTERM']:
        numeric(field, 0, 30, missing=[99])
    for field in ['LBO_REC', 'TBO_REC']:
        numeric(field, 1, 8, missing=[9], note='8 represents eight or more, not an exact count.')
    numeric('ILP_R', 0, 300, missing=[999], extras=[888], flag='F_MEDUC',
            note='000–003 plural delivery; 004–300 months; 888 no prior pregnancy is a real category. Do not treat 888 as an interval. Timing/availability must be reviewed in Stage 3.')
    numeric('PRECARE', 0, 10, missing=[99], flag='F_MPCB',
            note='00 means no prenatal care; eventual month/no-care status is time-dependent. Review in Stage 3.')
    numeric('PRECARE5', 1, 4, missing=[5], flag='F_MPCB', note='Recode of PRECARE; availability depends on prediction time.')
    numeric('PREVIS', 0, 98, missing=[99], flag='F_TPCV')
    numeric('PREVIS_REC', 1, 11, missing=[12], flag='F_TPCV')
    for i in range(4):
        numeric(f'CIG_{i}', 0, 98, missing=[99], flag=f'F_CIGS_{i}',
                note='98 represents 98 or more cigarettes daily.')
    numeric('M_Ht_In', 30, 78, missing=[99], flag='F_M_HT')
    numeric('BMI', 13.0, 69.9, missing=['99.9'], step=0.1, flag='F_M_HT')
    numeric('BMI_R', 1, 6, missing=[9], flag='F_M_HT',
            note='Guide has an apparent Obesity I label typo (35.0–34.9). No bin boundaries are inferred here; review before feature engineering.')
    numeric('PWgt_R', 75, 375, missing=[999], flag='F_PWGT')
    numeric('WTGAIN', 0, 98, missing=[99], flag='F_WTGAIN', note='98 represents 98 or more pounds gained.')
    numeric('WTGAIN_REC', 1, 5, missing=[9], flag='F_WTGAIN')
    numeric('RF_CESARN', 0, 30, missing=[99], flag='F_RF_NCESAR')
    numeric('PAY_REC', 1, 4, missing=[9], flag='F_PAY_REC',
            note='Payment for delivery; prenatal availability not established. Stage 3 must review before use.')
    numeric('DPLURAL', 1, 4, note='1 single; 2 twin; 3 triplet; 4 quadruplet or higher. Required singleton experiment later.')
    for field in ['COMBGEST', 'OEGest_Comb']:
        numeric(field, 17, 47, missing=[99], note='Target source only; never a predictor.')
    for field in ['GESTREC3', 'OEGest_R3']:
        numeric(field, 1, 2, missing=[3], note='1 <37 weeks; 2 >=37 weeks; 3 not stated -> NaN. Raw code counts retained; no rows dropped until Stage 2. Never a predictor.')
    flags = {'WIC':'F_WIC', 'RF_PDIAB':'F_RF_PDIAB', 'RF_GDIAB':'F_RF_GDIAB',
             'RF_PHYPE':'F_RF_PHYPER', 'RF_GHYPE':'F_RF_GHYPER', 'RF_EHYPE':'F_RF_ECLAMP',
             'RF_PPTERM':'F_RF_PPB', 'RF_INFTR':'f_RF_INFT', 'RF_FEDRG':'',
             'RF_ARTEC':'F_RF_INF_ART', 'RF_CESAR':'F_RF_CESAR', 'IP_GON':'F_IP_GONOR',
             'IP_SYPH':'F_IP_SYPH', 'IP_CHLAM':'F_IP_CHLAM', 'IP_HEPB':'F_IP_HEPATB',
             'IP_HEPC':'F_IP_HEPATC'}
    # RF_FEDRG's dedicated flag F_RF_INF_DRG (position 329) is absent
    # from the supplied layout. Do not infer a different flag or silently add a field.
    for field, flag in flags.items():
        structural = field in {'RF_FEDRG', 'RF_ARTEC'}
        specs[field] = dict(kind='categorical', allowed=['N','U','Y'] + (['X'] if structural else []),
                            missing=['U'], reporting_flag=flag,
                            semantic_note='X = not applicable; preserve as distinct category.' if structural else 'Y/N preserved; U -> missing.')
    specs['SEX'] = dict(kind='categorical', allowed=['F','M'], missing=[], reporting_flag='',
                        semantic_note='Recorded sex at birth; prenatal availability is not guaranteed. Review in Stage 3.')
    for row in selected.itertuples():
        if row.role == 'REPORTING_FLAG':
            specs[row.field] = dict(kind='numeric', allowed=['0','1'], missing=[], reporting_flag='',
                                    semantic_note='0 non-reporting; 1 reporting. Audit only, never a predictor; includes outcome reporting flags.')
    if set(specs) != set(selected.field):
        raise ValueError(f'Incomplete/extra rules: {set(selected.field) ^ set(specs)}')
    dictionary = selected.merge(evidence[['field','pdf_page','guide_position_line']], on='field', validate='one_to_one')
    dictionary['kind'] = dictionary.field.map(lambda f: specs[f]['kind'])
    dictionary['allowed_nonblank_codes_json'] = dictionary.field.map(lambda f: json.dumps(specs[f]['allowed']))
    dictionary['missing_codes_json'] = dictionary.field.map(lambda f: json.dumps([''] + specs[f]['missing']))
    dictionary['reporting_flag'] = dictionary.field.map(lambda f: specs[f]['reporting_flag'])
    dictionary['semantic_note'] = dictionary.field.map(lambda f: specs[f]['semantic_note'])
    dictionary['clean_storage'] = dictionary.field.map(lambda f: 'float64 with NaN' if specs[f]['kind']=='numeric' else 'string with null (NaN semantics)')
    dictionary['predictor_status'] = dictionary.role.map({'KEEP_candidate':'Candidate only; not yet approved',
                  'ABLATION_time_dependent':'Separate ablation only', 'TARGET':'Target audit only; excluded from predictors',
                  'REPORTING_FLAG':'Reporting audit only; excluded from predictors'})
    return dictionary, specs


def iter_selected_chunks(raw_path, selected, chunk_size):
    return pd.read_fwf(raw_path,
                       colspecs=list(zip(selected.pandas_colspec_start, selected.pandas_colspec_end)),
                       names=selected.field.tolist(), dtype=str, keep_default_na=False,
                       na_filter=False, encoding='ascii', chunksize=chunk_size)


def clean_chunk(chunk, specs):
    """Convert codes/blanks only; flag-zero policy awaits Stage 1 clarification."""
    result = {}
    for field in chunk.columns:
        rule = specs[field]
        values = chunk[field].str.strip()
        values = values.mask(values.isin([''] + rule['missing']))
        if rule['kind'] == 'numeric':
            result[field] = pd.to_numeric(values, errors='raise').astype('float64')
        else:
            result[field] = values.astype('string')
    return pd.DataFrame(result, index=chunk.index)


def parse_audit(raw_path, selected, specs, audit, output_dir, parquet_path, chunk_size):
    """Stream all selected fields; validate raw codes before cleaning/writing.

    Full selected-column cache is NOT the stratified modeling sample. No rows
    are removed, no binary target is constructed, and no model is fitted.
    A failed validation never publishes the temporary parquet as a usable cache.
    """
    started = time.perf_counter()
    before = raw_path.stat()
    if (before.st_size, before.st_mtime_ns) != (audit['raw_bytes'], audit['raw_mtime_ns']):
        raise RuntimeError('Raw file metadata changed since streaming audit.')
    columns = selected.field.tolist()
    counts = {field: Counter() for field in columns}
    missing = Counter()
    rows = 0
    first_raw = first_clean = None
    writer = None
    temporary = parquet_path.with_suffix('.parquet.partial')
    try:
        for chunk in iter_selected_chunks(raw_path, selected, chunk_size):
            invalid = []
            for field in columns:
                frequencies = chunk[field].str.strip().value_counts(dropna=False).to_dict()
                counts[field].update(frequencies)
                bad = set(frequencies) - set(specs[field]['allowed']) - {''}
                invalid.extend((field, token, int(frequencies[token])) for token in bad)
            if invalid:
                pd.DataFrame(invalid, columns=['field','invalid_code','count_in_chunk']).to_csv(output_dir/'invalid_codes.csv',index=False)
                raise ValueError(f'Out-of-guide codes after {rows} parsed rows: {invalid[:12]}. STOP and investigate alignment.')
            clean = clean_chunk(chunk, specs)
            missing.update({f:int(v) for f,v in clean.isna().sum().items()})
            if first_raw is None:
                first_raw, first_clean = chunk.head(5).copy(), clean.head(5).copy()
            table = pa.Table.from_pandas(clean, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table)
            rows += len(chunk)
            if rows % (chunk_size * 10) == 0:
                print(f'Validated and cached {rows:,} records', flush=True)
    finally:
        if writer is not None:
            writer.close()
    if rows != audit['raw_rows']:
        raise ValueError(f'Parsed/raw row count mismatch: {rows} vs {audit["raw_rows"]}')
    after = raw_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Raw source changed during parsing.')
    temporary.replace(parquet_path)
    value_rows = [dict(field=f, raw_code=code, count=int(n)) for f in columns for code,n in sorted(counts[f].items())]
    pd.DataFrame(value_rows).to_csv(output_dir/'raw_code_counts.csv', index=False)
    missing_table = pd.DataFrame([dict(field=f, role=selected.set_index('field').loc[f,'role'],
            raw_rows=rows, raw_blank_count=int(counts[f].get('',0)),
            unknown_code_count=sum(counts[f].get(c,0) for c in specs[f]['missing']),
            missing_count=int(missing[f]), missing_pct=100*missing[f]/rows,
            observed_nonblank_codes=len(set(counts[f])-{''}), invalid_code_count=0) for f in columns])
    missing_table.to_csv(output_dir/'missingness.csv',index=False)
    first_raw.to_csv(output_dir/'parsed_sample_raw_codes.csv',index=False)
    first_clean.to_csv(output_dir/'parsed_sample_clean.csv',index=False)
    # Report flag/value combinations without making undocumented replacements.
    flag_rows = []
    for f in columns:
        flag = specs[f]['reporting_flag']
        if flag:
            flag_rows.append(dict(field=f, flag=flag,
                flag_nonreporting_rows=int(counts[flag].get('0',0)),
                flag_blank_rows=int(counts[flag].get('',0)),
                field_blank_rows=int(counts[f].get('',0))))
    pd.DataFrame(flag_rows).to_csv(output_dir/'reporting_flags.csv',index=False)
    result = dict(parsed_rows=rows, selected_columns=len(columns),
                  role_counts=selected.role.value_counts().to_dict(), invalid_code_count=0,
                  chunk_size=chunk_size, parquet_path=str(parquet_path),
                  parquet_bytes=parquet_path.stat().st_size,
                  elapsed_seconds=time.perf_counter()-started,
                  raw_sha256=audit['sha256'], raw_mtime_ns=after.st_mtime_ns,
                  rows_dropped=0, binary_target_created=False, modeling_sample_created=False)
    (output_dir/'parse_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return missing_table, first_clean, result
