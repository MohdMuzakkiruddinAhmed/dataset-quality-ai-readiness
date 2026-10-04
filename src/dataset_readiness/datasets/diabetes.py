"""Pinned UCI Diabetes source, explicit eligibility and natural-missing reference."""

import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from dataset_readiness.artifacts import csv_bytes, sha256, write_once
from dataset_readiness.datasets.tasks import BinaryTask

SOURCE_URL = 'https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip'
METADATA_URL = 'https://archive.ics.uci.edu/api/dataset?id=296'
SOURCE_FILES = {
    'source.zip': 'f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a',
    'diabetic_data.csv': '0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97',
    'IDS_mapping.csv': 'f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e',
    'uci_metadata.json': '6e6e0934d175afe24f54f2199f9895eab21844205fe46be7970c106a104725e9',
}
REFERENCE_VERSION = 'eligible_encounters_natural_missing_v1'
EXCLUDED_DISPOSITIONS = ('11', '13', '14', '19', '20', '21')
SOURCE_COLUMNS = (
    'encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code',
    'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
    'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
    'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted',
)
NUMERIC = ('time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
           'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses')
QUESTION_MARK_COLUMNS = ('race', 'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3')
MAPPING_COLUMNS = ('admission_type_id', 'discharge_disposition_id', 'admission_source_id')
DIABETES = BinaryTask('diabetes', 'diabetes_readmission_30d_v1', SOURCE_COLUMNS[2:-1], NUMERIC,
                      'readmitted_30d', 'NOT_EARLY', 'EARLY', ('encounter_id', 'patient_nbr', 'readmitted', 'source_row_offset'))


def source_provenance():
    return {'dataset': 'Diabetes 130-US Hospitals for Years 1999-2008', 'uci_id': 296,
            'repository_url': 'https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008',
            'source_url': SOURCE_URL, 'metadata_url': METADATA_URL, 'source_sha256': dict(SOURCE_FILES),
            'citation': 'Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008. UCI Machine Learning Repository.',
            'doi': '10.24432/C5230J', 'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
            'license_basis': 'UCI repository license section checked on 2026-09-30',
            'download_variant': 'Original archive diabetic_data.csv; not the separately transformed API data.csv'}


def download_diabetes(directory):
    directory = Path(directory)
    def verified(name, payload):
        if sha256(payload) != SOURCE_FILES[name]: raise ValueError('Source checksum mismatch: ' + name)
        write_once(directory/name, payload)
        return payload
    if (directory/'source.zip').exists(): archive = verified('source.zip', (directory/'source.zip').read_bytes())
    else:
        with urlopen(SOURCE_URL, timeout=45) as response: archive = verified('source.zip', response.read())
    with ZipFile(BytesIO(archive)) as zipped:
        if set(zipped.namelist()) != {'diabetic_data.csv', 'IDS_mapping.csv'}: raise ValueError('Unexpected source archive members')
        for name in ('diabetic_data.csv', 'IDS_mapping.csv'):
            # Read only named members, never extract arbitrary archive paths.
            body = (directory/name).read_bytes() if (directory/name).exists() else zipped.read(name)
            verified(name, body)
    if (directory/'uci_metadata.json').exists(): body = (directory/'uci_metadata.json').read_bytes()
    else:
        with urlopen(METADATA_URL, timeout=45) as response: body = response.read()
    verified('uci_metadata.json', body)
    metadata = json.loads(body)['data']
    if metadata['uci_id'] != 296 or [v['name'] for v in metadata['variables']] != list(SOURCE_COLUMNS):
        raise ValueError('UCI documentation differs from the pinned schema')
    return dict(SOURCE_FILES)


def parse_mapping(payload):
    sections = {}; current = None
    for row in csv.reader(StringIO(payload.decode('utf8'))):
        if not row or not any(row): continue
        if len(row) != 2: raise ValueError('Invalid mapping row')
        if row[0] in MAPPING_COLUMNS and row[1] == 'description':
            if row[0] in sections: raise ValueError('Duplicate mapping section')
            current = row[0]; sections[current] = {}; continue
        if current is None or not re.fullmatch(r'[1-9][0-9]*', row[0]) or row[0] in sections[current]:
            raise ValueError('Invalid or duplicate mapping code')
        sections[current][row[0]] = row[1].strip()
    if set(sections) != set(MAPPING_COLUMNS): raise ValueError('Incomplete ID mapping')
    return sections


def parse_source(payload, mappings):
    text = payload.decode('utf8'); header = next(csv.reader(StringIO(text)), [])
    if tuple(header) != SOURCE_COLUMNS: raise ValueError('Unexpected Diabetes source header/order')
    raw = pd.read_csv(BytesIO(payload), dtype=str, keep_default_na=False)
    if not raw.index.equals(pd.RangeIndex(len(raw))): raise ValueError('Malformed source row width')
    if raw.empty or raw.isna().any().any() or raw.eq('').any().any(): raise ValueError('Empty/malformed source fields')
    if not raw.readmitted.isin(['<30', '>30', 'NO']).all(): raise ValueError('Unknown or missing readmission outcome')
    for col in ('encounter_id', 'patient_nbr', *NUMERIC):
        if not raw[col].str.fullmatch(r'0|[1-9][0-9]*').all(): raise ValueError('Noncanonical nonnegative integer: ' + col)
        values = [int(v) for v in raw[col]]
        if any(v > 2**63-1 for v in values): raise ValueError('Integer overflow: ' + col)
        if col in ('encounter_id', 'patient_nbr') and any(v == 0 for v in values): raise ValueError('IDs must be positive')
    if raw.encounter_id.duplicated().any(): raise ValueError('Duplicate encounter ID')
    if not raw.time_in_hospital.astype(int).between(1,14).all(): raise ValueError('Invalid length of stay')
    for col in MAPPING_COLUMNS:
        if not raw[col].isin(mappings[col]).all(): raise ValueError('Unmapped source ID code: ' + col)
    for col in raw:
        if col not in QUESTION_MARK_COLUMNS and raw[col].eq('?').any(): raise ValueError('Unexpected missing marker: ' + col)
    return raw


def natural_missingness(frame):
    return {'scope': 'observed reference missingness; not injected ground truth',
            'rows': len(frame), 'feature_cells': len(frame)*len(DIABETES.features),
            'missing_cells': int(frame[list(DIABETES.features)].isna().sum().sum()),
            'rows_by_column': {c: [int(i) for i in frame.index[frame[c].isna()]] for c in DIABETES.features}}


def build_reference(directory):
    directory = Path(directory); mappings = parse_mapping((directory/'IDS_mapping.csv').read_bytes())
    raw = parse_source((directory/'diabetic_data.csv').read_bytes(), mappings)
    eligible = ~raw.discharge_disposition_id.isin(EXCLUDED_DISPOSITIONS)
    selected = raw.loc[eligible].copy()
    ids = pd.Index(selected.encounter_id.astype('int64'), name='row_id')
    frame = selected.loc[:, list(DIABETES.features)].copy(); frame.index = ids
    for col in DIABETES.numeric: frame[col] = frame[col].astype('Int64')
    for col in QUESTION_MARK_COLUMNS: frame[col] = frame[col].mask(frame[col].eq('?'), np.nan)
    frame[DIABETES.target] = np.where(selected.readmitted.eq('<30'), 'EARLY', 'NOT_EARLY')
    lineage = pd.DataFrame({'patient_nbr': selected.patient_nbr.astype('int64').to_numpy(),
                            'source_row_offset': selected.index.to_numpy(), 'source_readmitted': selected.readmitted.to_numpy()}, index=ids)
    frame, lineage = frame.sort_index(), lineage.sort_index()
    DIABETES.validate_frame(frame, training=True)
    counts = lambda series: {str(k): int(v) for k,v in series.value_counts().sort_index().items()}
    audit = {'policy': REFERENCE_VERSION, 'raw_rows': len(raw), 'raw_patients': int(raw.patient_nbr.nunique()),
             'reference_rows': len(frame), 'reference_patients': int(lineage.patient_nbr.nunique()),
             'excluded_rows': int((~eligible).sum()), 'excluded_encounter_ids': [int(i) for i in raw.loc[~eligible,'encounter_id']],
             'excluded_dispositions': {code: {'description': mappings['discharge_disposition_id'][code],
                                            'count': int(raw.discharge_disposition_id.eq(code).sum())} for code in EXCLUDED_DISPOSITIONS},
             'eligibility_reason': 'Exclude explicit expired/hospice discharge codes; retain all other documented encounters. Not a reproduction of the original paper cohort.',
             'raw_outcomes': counts(raw.readmitted), 'retained_source_outcomes': counts(selected.readmitted),
             'reference_class_counts': counts(frame[DIABETES.target]),
             'raw_question_marks': {c: int(raw[c].eq('?').sum()) for c in DIABETES.features},
             'reference_missing_by_column': {c: int(frame[c].isna().sum()) for c in DIABETES.features},
             'literal_none_retained': {c: int(frame[c].eq('None').sum()) for c in ('max_glu_serum','A1Cresult')},
             'duplicate_encounter_ids': 0, 'duplicate_predictor_rows': int(frame.duplicated(list(DIABETES.features)).sum()),
             'constant_predictors': [c for c in DIABETES.features if frame[c].nunique(dropna=False)==1],
             'row_id_policy': 'Original encounter_id, sorted; patient ID and source outcome/offset remain in evaluator lineage only',
             'target_mapping': {'<30':'EARLY','>30':'NOT_EARLY','NO':'NOT_EARLY'},
             'missing_policy': 'Only documented ? markers become missing; literal None, Unknown/Invalid and mapping codes remain categories. No imputation, complete-case filtering or feature dropping.',
             'prediction_time': 'At discharge; all 47 documented source features, excluding IDs and raw readmission outcome',
             'limitations': ['NO denotes no recorded readmission, not verified complete follow-up; exactly day 30 is not specified by the source label definitions.',
                             'Multiple encounters per patient retained; encounter-weighted scores and patient-disjoint random holdout, not time/hospital-external validation.',
                             'Natural missingness and constant features remain; structurally valid reference is not verified semantic ground truth.',
                             'Discharge-time predictors include care decisions and demographics; no causal, prospective deployment or fairness claim.']}
    return frame, lineage, audit


def read_reference(payload):
    frame = pd.read_csv(BytesIO(payload), index_col='row_id', keep_default_na=False, na_values=[''],
                        dtype={'row_id':'int64', **{c:'Int64' for c in DIABETES.numeric},
                               **{c:object for c in (*DIABETES.categorical,DIABETES.target)}})
    DIABETES.validate_frame(frame)
    if csv_bytes(frame) != payload: raise ValueError('Diabetes reference CSV does not round-trip')
    return frame
