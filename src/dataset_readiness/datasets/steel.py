"""Pinned UCI Steel reference with conservative within-day lag lineage."""

from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from dataset_readiness.artifacts import sha256, write_once
from dataset_readiness.datasets.multiclass import MulticlassTask

REFERENCE_VERSION = 'steel_within_day_lag15_v1'
SOURCE_CSV = 'Steel_industry_data.csv'
URLS = {'source.zip': 'https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip',
        'uci_metadata.json': 'https://archive.ics.uci.edu/api/dataset?id=851'}
SOURCE_SHA256 = {
    'source.zip': 'd82d28b33780ff1582507fcf08ae764ff648af459d58234370c551e62aadeaef',
    'uci_metadata.json': 'eb003eacd9268841c144198c78c7f797127aee5686656938d9c13c63427f1475',
    SOURCE_CSV: '9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc',
}
SENSORS = ('Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
           'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor')
COLUMNS = ('date', *SENSORS[:3], 'CO2(tCO2)', *SENSORS[3:], 'NSM', 'WeekStatus', 'Day_of_week', 'Load_Type')
LAGS = tuple('lag15_' + c for c in SENSORS)
STEEL = MulticlassTask('steel', 'steel_load_type_lag15_v1', (*LAGS, 'NSM', 'WeekStatus', 'Day_of_week'),
    (*LAGS, 'NSM'), 'Load_Type', ('Light_Load', 'Medium_Load', 'Maximum_Load'),
    ('date', 'row_id', 'source_row_offset', 'lag_source_row_id', 'CO2(tCO2)', *SENSORS))
SPLIT = {'protocol': 'chronological_2018_q4_v1', 'train_start': '2018-01-01 00:00:00',
         'test_start': '2018-10-01 00:00:00', 'test_end_exclusive': '2019-01-01 00:00:00'}
POLICY = {
    'task': 'Predict recorded Load_Type at timestamp t from calendar at t and five sensor readings at t minus 15 minutes',
    'target_basis': 'UCI metadata declares categorical Load_Type target; repository associated task and original paper concern regression',
    'interpretation': 'New recorded-class classification benchmark; not a reproduction of energy-consumption regression or a validated operational forecast',
    'availability_assumption': 'Previous timestamp sensor readings are available before the target interval; source does not establish measurement latency or interval start/end semantics',
    'label_limitation': 'Operational construction of Load_Type is not established by the retrieved source documentation; labels may reflect operating schedules',
    'time_zone': 'Source naive local timestamps; no timezone localization or UTC conversion inferred',
    'exclusions': 'Exclude all 00:00 records, then exclude targets without a same-day exactly 15-minute earlier nonmidnight record; never shift source dates',
    'lag_scope': 'Within-day only; no previous label, current sensor or CO2 feature; earlier held-out sensor readings are allowed as rolling observed covariates, with no retraining',
    'duplicate_policy': 'Preserve repeated measurements at distinct unique timestamps; reject ambiguous duplicate timestamps',
    'missing_policy': 'Preserve empty numeric sensor observations as NaN; reject missing timestamps, calendar or target and unknown tokens; no complete-case filtering',
    'split': 'Fixed January-September training, October-December holdout; all classes required in both; no outcome-based split changes',
}


def provenance():
    return {'dataset': 'UCI Steel Industry Energy Consumption', 'uci_id': 851,
        'citation': 'V E, S., Shin, C., & Cho, Y. (2021). Steel Industry Energy Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52G8C.',
        'page': 'https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption',
        'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
        'paper_abstract': 'https://www.tandfonline.com/doi/abs/10.1080/09613218.2020.1809983',
        'urls': URLS, 'source_sha256': SOURCE_SHA256, 'transformations': POLICY}


def download_steel(directory):
    directory = Path(directory)
    for name, url in URLS.items():
        path = directory / name
        if path.exists(): payload = path.read_bytes()
        else:
            with urlopen(url, timeout=60) as response: payload = response.read()
        if sha256(payload) != SOURCE_SHA256[name]: raise ValueError('Pinned Steel source hash mismatch: ' + name)
        write_once(path, payload)
    with ZipFile(BytesIO((directory / 'source.zip').read_bytes())) as archive:
        payload = archive.read(SOURCE_CSV)
    if sha256(payload) != SOURCE_SHA256[SOURCE_CSV]: raise ValueError('Pinned Steel CSV hash mismatch')
    write_once(directory / SOURCE_CSV, payload)
    return dict(SOURCE_SHA256)


def parse_source(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'uci_metadata.json').read_bytes())['data']
    if (metadata['uci_id'] != 851 or tuple(v['name'] for v in metadata['variables']) != COLUMNS
            or [(v['name'], v['type']) for v in metadata['variables'] if v['role'] == 'Target'] != [('Load_Type', 'Categorical')]):
        raise ValueError('Unexpected UCI Steel metadata/schema')
    raw = pd.read_csv(directory / SOURCE_CSV, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if tuple(raw.columns) != COLUMNS or raw.empty: raise ValueError('Unexpected Steel CSV schema')
    raw.index = pd.Index(range(len(raw)), name='row_id')
    stamp = pd.to_datetime(raw.date, format='%d/%m/%Y %H:%M', errors='raise', exact=True)
    if (stamp.isna().any() or not stamp.is_unique or not stamp.dt.year.eq(2018).all()
            or not stamp.dt.minute.isin([0, 15, 30, 45]).all()
            or not stamp.dt.strftime('%d/%m/%Y %H:%M').eq(raw.date).all()):
        raise ValueError('Ambiguous, out-of-year or off-grid source timestamps')
    parsed = raw.copy()
    for c in (*SENSORS, 'CO2(tCO2)', 'NSM'):
        # Python float preserves correct binary64 rounding of each source token.
        # Only literal empty tokens are missing; textual NaN/Inf are malformed.
        values = [float(v) if v != '' else np.nan for v in raw[c]]
        if any(not np.isfinite(v) for v, token in zip(values, raw[c]) if token != ''):
            raise ValueError('Nonfinite source value: ' + c)
        parsed[c] = pd.Series(values, index=raw.index, dtype=float)
    seconds = stamp.dt.hour * 3600 + stamp.dt.minute * 60
    if not parsed.NSM.eq(seconds).all(): raise ValueError('NSM does not agree with literal date')
    parsed.NSM = parsed.NSM.astype('int64')
    weekdays = stamp.dt.day_name()
    if not raw.Day_of_week.eq(weekdays).all(): raise ValueError('Day_of_week mismatch')
    if not raw.WeekStatus.eq(np.where(stamp.dt.dayofweek >= 5, 'Weekend', 'Weekday')).all():
        raise ValueError('WeekStatus mismatch')
    if not raw.Load_Type.isin(STEEL.labels).all(): raise ValueError('Unknown or missing source target')
    return raw, parsed, stamp


def build_reference(directory):
    raw, parsed, stamp = parse_source(directory)
    ordered = parsed.assign(timestamp=stamp).sort_values('timestamp', kind='stable')
    midnight_ids = ordered.index[ordered.NSM.eq(0)].tolist()
    safe = ordered.loc[ordered.NSM.ne(0)].copy()
    preceding = safe.shift(1)
    eligible = ((safe.timestamp - preceding.timestamp).eq(pd.Timedelta(minutes=15))
                & safe.timestamp.dt.normalize().eq(preceding.timestamp.dt.normalize()))
    selected = safe.loc[eligible]
    reference = pd.DataFrame(index=selected.index)
    for original, lag in zip(SENSORS, LAGS): reference[lag] = preceding.loc[eligible, original].astype(float)
    for c in ('NSM', 'WeekStatus', 'Day_of_week', 'Load_Type'): reference[c] = selected[c]
    prior_ids = pd.Series(safe.index, index=safe.index).shift(1).loc[eligible].astype('int64')
    lineage = pd.DataFrame({'timestamp': selected.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S'),
        'lag_timestamp': preceding.loc[eligible, 'timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'lag_source_row_id': prior_ids}, index=reference.index)
    STEEL.validate_frame(reference, training=True)
    audit = {'raw_rows': len(raw), 'reference_rows': len(reference),
        'excluded_midnight_row_ids': midnight_ids, 'excluded_no_prior_interval_row_ids': safe.index[~eligible].tolist(),
        'excluded_rows': len(raw) - len(reference), 'raw_timestamp_order_monotonic': bool(stamp.is_monotonic_increasing),
        'duplicate_timestamps': int(stamp.duplicated().sum()),
        'repeated_records_excluding_date': int(raw.duplicated(subset=list(COLUMNS[1:])).sum()),
        'raw_empty_by_column': {c: int(raw[c].eq('').sum()) for c in COLUMNS},
        'reference_missing_by_column': {c: int(reference[c].isna().sum()) for c in STEEL.features},
        'raw_class_counts': {c: int(raw.Load_Type.eq(c).sum()) for c in STEEL.labels},
        'reference_class_counts': {c: int(reference.Load_Type.eq(c).sum()) for c in STEEL.labels},
        'policy': POLICY}
    return reference, lineage, audit


def chronological_split(reference, lineage, config):
    if config != SPLIT: raise ValueError('Unsupported Steel chronological split')
    STEEL.validate_frame(reference, training=True)
    if not reference.index.equals(lineage.index): raise ValueError('Lineage row order mismatch')
    times = pd.to_datetime(lineage.timestamp, format='%Y-%m-%d %H:%M:%S')
    lag_times = pd.to_datetime(lineage.lag_timestamp, format='%Y-%m-%d %H:%M:%S')
    if (not times.is_monotonic_increasing or not times.is_unique or
            not (times - lag_times).eq(pd.Timedelta(minutes=15)).all()
            or not times.dt.normalize().eq(lag_times.dt.normalize()).all()):
        raise ValueError('Invalid time ordering or lag lineage')
    train = reference.loc[(times >= config['train_start']) & (times < config['test_start'])].copy()
    test = reference.loc[(times >= config['test_start']) & (times < config['test_end_exclusive'])].copy()
    if len(train) + len(test) != len(reference): raise ValueError('Rows outside declared time window')
    STEEL.validate_frame(train, training=True); STEEL.validate_frame(test, training=True)
    if set(train.index) & set(test.index): raise ValueError('Chronological partitions overlap')
    return train, test
