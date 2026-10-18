"""Stage 1 follow-up: quantify flag=0/nonmissing conflicts without changing data."""
from pathlib import Path
import argparse
import json

import pandas as pd
import pyarrow.parquet as pq


def audit(root: Path):
    dictionary = pd.read_csv(root/'reports/tables/stage1_data_dictionary.csv', keep_default_na=False)
    cache = root/'data/processed/natality2024_selected_full.parquet'
    output = root/'reports/tables/stage1_reporting_flag_conflicts.csv'
    pairs = [(r.field, r.reporting_flag) for r in dictionary.itertuples() if r.reporting_flag]
    totals = {field: dict(field=field, reporting_flag=flag, rows=0,
                         code_missing_count=0, flag_zero_count=0,
                         flag_zero_nonmissing_count=0, proposed_missing_count=0)
              for field, flag in pairs}
    columns = list(dict.fromkeys(value for pair in pairs for value in pair))
    for batch in pq.ParquetFile(cache).iter_batches(batch_size=25_000, columns=columns):
        frame = batch.to_pandas()
        for field, flag in pairs:
            missing = frame[field].isna()
            zero = frame[flag].eq(0)
            row = totals[field]
            row['rows'] += len(frame)
            row['code_missing_count'] += int(missing.sum())
            row['flag_zero_count'] += int(zero.sum())
            row['flag_zero_nonmissing_count'] += int((zero & ~missing).sum())
            row['proposed_missing_count'] += int((zero | missing).sum())
    result = pd.DataFrame(totals.values())
    result['code_missing_pct'] = 100*result.code_missing_count/result.rows
    result['proposed_missing_pct'] = 100*result.proposed_missing_count/result.rows
    result.to_csv(output,index=False)
    print(result.to_string(index=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.root.resolve())
