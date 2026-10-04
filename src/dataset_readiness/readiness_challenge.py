"""Prospectively fixed six-case observation, combined repair and label policy."""
from collections import Counter
from hashlib import sha256 as digest
from io import BytesIO
import json
import math
import numpy as np
import pandas as pd
from dataset_readiness.agents.observed_task import validate_observation
from dataset_readiness.artifacts import csv_bytes, json_bytes, sha256
from dataset_readiness.datasets.diabetes import DIABETES as TASK

VERSION = 'readiness_challenge_v1'
CASES = (
    {'id':'C01','family':'native','seed':None,'severity':0},
    {'id':'C02','family':'predictor_missingness','seed':2001,'severity':.05},
    {'id':'C03','family':'numeric_domain_negatives','seed':2001,'severity':.1},
    {'id':'C04','family':'numeric_domain_negatives','seed':2002,'severity':.2},
    {'id':'C05','family':'all_predictors_missing','seed':2001,'severity':1},
    {'id':'C06','family':'all_training_targets_missing','seed':2001,'severity':1},
)
METRICS = ('macro_f1','roc_auc','average_precision')
LABELS = ('READY','REMEDIATE','NOT_READY')
EPSILON = 1e-12
POLICY = {
    'version': VERSION,
    'sampling':'Sorted row IDs, task column order, row-major eligible coordinates; numpy default_rng(seed).choice without replacement, sorted selected offsets; floor(rate * eligible_count).',
    'missing_eligibility':'Observed nonmissing predictor cells, including categorical and numeric values; native missing cells excluded.',
    'domain_eligibility':'Observed strictly positive integer cells in the eight declared numeric predictors; zero excluded because exact arithmetic negation cannot make zero negative.',
    'domain_operation':'Exact arithmetic negation x -> -x; no inverse repair.',
    'all_missing':'All predictors or all targets masked; no sampling draw at 100 percent; seed identifies case only.',
    'serialization':'UTF-8 LF CSV with row_id; empty alone denotes missing; observation numerics nullable Int64; repaired numerics nullable Float64; literal None category retained.',
    'repair':'Mask supported negative numeric cells first; then fill only supported original missing cells and those newly masked cells with train-only nonnegative numeric median or categorical mode.',
    'numeric_donors':'All nonmissing nonnegative values in the same observed training column after approved masks; unapproved negatives never donors.',
    'numeric_empty_fallback':0.0,
    'categorical_empty_fallback':'__MISSING__',
    'categorical_tie':'Lexicographically smallest string among modes.',
    'protected':'All targets, row IDs/order, column order, and every predictor outside approved coordinates; numerical dtype promotion changes representation only.',
    'rules_readiness':'NOT_READY for unusable targets or all predictors missing; REMEDIATE for any predictor missingness or declared numeric violation; READY otherwise. No canonical scores consulted.',
    'critical':'Unusable target, any declared numeric violation, or all predictor cells missing.',
    'label_tolerance':.02,'severe_loss':.10,'minimum_recovery':.50,'epsilon':EPSILON,
    'class_coverage':'At least two distinct determinate cases in each of READY, REMEDIATE, NOT_READY; otherwise inconclusive without replacements.',
    'scalar':'Existing task_suitability is categorical; correlation not estimable; no substitute scalar or manufactured probability vector.',
}




def read_frame(payload, *, repaired=False):
    dtype = 'Float64' if repaired else 'Int64'
    frame = pd.read_csv(BytesIO(payload),index_col='row_id',keep_default_na=False,na_values=[''],
        dtype={'row_id':'int64',**{c:dtype for c in TASK.numeric},**{c:object for c in (*TASK.categorical,TASK.target)}},float_precision='round_trip')
    validate_observation(frame,TASK)
    if csv_bytes(frame)!=payload: raise ValueError('Noncanonical challenge CSV')
    return frame


def critical(frame):
    labels=frame[TASK.target]
    return {'unusable_target':bool(labels.isna().any() or not labels.isin([TASK.negative_label,TASK.positive_label]).all() or labels.nunique()!=2),
        'numeric_domain_violation':bool(any(frame[c].lt(0).fillna(False).any() for c in TASK.numeric)),
        'all_predictors_unavailable':bool(frame[list(TASK.features)].isna().all().all())}


def rules_readiness(frame):
    validate_observation(frame,TASK); flags=critical(frame)
    if flags['unusable_target'] or flags['all_predictors_unavailable']: return 'NOT_READY'
    if flags['numeric_domain_violation'] or frame[list(TASK.features)].isna().any().any(): return 'REMEDIATE'
    return 'READY'


def generate(original, case):
    if case not in CASES: raise ValueError('Outside the six frozen cases')
    TASK.validate_frame(original,training=True)
    if not original.index.is_monotonic_increasing: raise ValueError('Sorted row IDs required')
    if any(str(original[c].dtype)!='Int64' for c in TASK.numeric): raise ValueError('Integer observation required')
    before=csv_bytes(original); observed=original.copy(deep=True); family=case['family']
    columns=list(TASK.numeric if family=='numeric_domain_negatives' else [TASK.target] if family=='all_training_targets_missing' else TASK.features)
    values=original[columns]
    eligible = values.gt(0).fillna(False).to_numpy(dtype=bool) if family=='numeric_domain_negatives' else values.notna().to_numpy()
    offsets=np.flatnonzero(eligible.reshape(-1))
    if family=='native': selected=np.array([],dtype=int)
    elif family.startswith('all_'): selected=offsets
    else:
        count=math.floor(case['severity']*len(offsets))
        selected=np.sort(np.random.default_rng(case['seed']).choice(offsets,size=count,replace=False))
    selected_rows,selected_cols=np.divmod(selected,len(columns)); coordinates={}
    for j,c in enumerate(columns):
        rows=original.index[selected_rows[selected_cols==j]]
        if not len(rows): continue
        if family=='numeric_domain_negatives': observed.loc[rows,c]=-original.loc[rows,c]
        else: observed.loc[rows,c]=pd.NA if c in TASK.numeric else np.nan
        coordinates[c]=[int(r) for r in rows]
    validate_observation(observed,TASK)
    if family!='all_training_targets_missing' and not original[TASK.target].equals(observed[TASK.target]): raise ValueError('Target changed')
    assert csv_bytes(original)==before
    return observed,{'version':VERSION,'case':case,'policy_sha256':sha256(json_bytes(POLICY)),
        'reference_sha256':sha256(before),'observed_sha256':sha256(csv_bytes(observed)),
        'eligible_cells':len(offsets),'changed_cells':len(selected),'rows_by_column':coordinates}


def _selection(kind,column,rows):
    return 'selection_'+digest(json.dumps([kind,column,tuple(rows)]).encode()).hexdigest()[:16]


def approved_coordinates(frame, assessment=None):
    """None is the independent canonical rules path, never an agent fallback."""
    negative={c:tuple(int(r) for r in frame.index[frame[c].lt(0).fillna(False)]) for c in TASK.numeric}
    missing={c:tuple(int(r) for r in frame.index[frame[c].isna()]) for c in TASK.features}
    if assessment is None: return negative,missing,[]
    if set(assessment.resolved)!={i['issue_id'] for i in assessment.report['issues']}: raise ValueError('Resolution mismatch')
    masks={c:set() for c in TASK.numeric};fills={c:set() for c in TASK.features};rejected=[]
    for issue in assessment.report['issues']:
        identifier=issue['issue_id'];c=issue['column'];kind=issue['issue_type'];action=issue['recommended_action']
        rows=tuple(assessment.resolved[identifier]);expected=None;target=None
        if c in negative and kind=='range_violation' and action=='REPLACE_INVALID_WITH_NAN': expected=negative[c];target=masks
        if c in missing and kind=='missing_values' and action==('IMPUTE_NUMERIC_MEDIAN' if c in TASK.numeric else 'IMPUTE_CATEGORICAL_MODE'): expected=missing[c];target=fills
        if expected and issue['level']=='cell' and not issue['row_ids'] and rows==expected and issue['selection_id']==_selection(kind,c,expected):
            target[c].update(rows)
        else: rejected.append({'issue_id':identifier,'reason':'Unsupported action, coarse/partial/unvalidated selection, empty selection, or protected target'})
    return {c:tuple(sorted(v)) for c,v in masks.items()},{c:tuple(sorted(v)) for c,v in fills.items()},rejected


def repair(frame, assessment=None):
    validate_observation(frame,TASK);before=csv_bytes(frame)
    masks,fills,rejected=approved_coordinates(frame,assessment)
    after=frame.copy(deep=True)
    for c in TASK.numeric: after[c]=after[c].astype('Float64')
    steps=[]
    for c in TASK.features:
        domain=list(masks.get(c,()));original_missing=list(fills[c]);rows=sorted(set(domain)|set(original_missing))
        if not rows: continue
        if domain: after.loc[domain,c]=pd.NA
        if c in TASK.numeric:
            donors=after[c].dropna();donors=donors[donors.ge(0)]
            fill=float(donors.median()) if len(donors) else 0.0
        else:
            donors=after[c].dropna();counts=Counter(donors)
            fill=min(k for k,v in counts.items() if v==max(counts.values())) if counts else '__MISSING__'
        after.loc[rows,c]=fill
        steps.append({'column':c,'domain_rows':domain,'missing_rows':original_missing,'fill_rows':rows,'fill_value':fill,'donor_count':len(donors)})
    validate_observation(after,TASK)
    if not frame.index.identical(after.index) or not frame.columns.equals(after.columns) or not frame[TASK.target].equals(after[TASK.target]): raise ValueError('Protected structure or target changed')
    edits=0
    for c in TASK.features:
        equal=frame[c].eq(after[c]).fillna(False)|(frame[c].isna()&after[c].isna())
        actual=set(int(r) for r in frame.index[~equal]);expected=set(masks.get(c,()))|set(fills[c])
        if actual!=expected: raise ValueError('Unexpected or omitted edits')
        edits+=len(actual)
    if csv_bytes(frame)!=before: raise ValueError('Repair mutated source')
    manifest={'version':VERSION,'input_sha256':sha256(before),'repaired_sha256':sha256(csv_bytes(after)),
        'canonical':assessment is None,'steps':steps,'rejected':rejected,'changed_cells':edits,
        'assessment_sha256':None if assessment is None else sha256(json_bytes({'report':assessment.report,'resolved':assessment.resolved}))}
    verification={'status':'passed','protected_target_and_structure':True,'unapproved_value_changes':0,
        'numeric_dtype':'Float64','critical_before':critical(frame),'critical_after':critical(after),'changed_cells':edits}
    return after,manifest,verification


def finite(value): return type(value) in (int,float) and math.isfinite(value)


def label(reference, observed, repaired, *, before_critical, after_critical, safe, data_failure=False):
    """Ordered Stage 45 policy; execution failures are never data failures."""
    result={'label':'ABSTAIN','reason':'undefined_required_metrics','effects':{}}
    if before_critical['unusable_target'] or data_failure:
        return {**result,'label':'NOT_READY','reason':'unusable_supervised_target_or_verified_data_failure'}
    for name in METRICS:
        ref,obs,rep=(v.get(name) for v in (reference or {},observed or {},repaired or {}))
        loss=ref-obs if finite(ref) and finite(obs) else None
        residual=ref-rep if finite(ref) and finite(rep) else None
        benefit=rep-obs if finite(rep) and finite(obs) else None
        result['effects'][name]={'loss':loss,'residual_loss':residual,'benefit':benefit,
            'recovery':benefit/loss if benefit is not None and loss is not None and loss>EPSILON else None}
    values=list(result['effects'].values())
    if any(v['loss'] is None for v in values): return result
    if not any(before_critical.values()) and all(v['loss']<=.02+EPSILON for v in values):
        return {**result,'label':'READY','reason':'all_signed_losses_within_tolerance_no_critical_issue'}
    # A missing repair score leaves the repair acceptability test unresolved.
    if any(v['residual_loss'] is None for v in values) or safe is None or after_critical is None: return result
    acceptable=safe and not any(after_critical.values()) and all(v['residual_loss']<=.02+EPSILON and
        (v['loss']<=.02+EPSILON or v['recovery'] is not None and v['recovery']>=.50-EPSILON) for v in values)
    if acceptable: return {**result,'label':'REMEDIATE','reason':'safe_canonical_repair_meets_residual_and_recovery_limits'}
    if any(v['loss']>=.10-EPSILON for v in values):
        return {**result,'label':'NOT_READY','reason':'severe_loss_without_acceptable_canonical_repair'}
    return {**result,'reason':'material_or_critical_condition_without_determinate_utility_label'}


def readiness_metrics(truth, predictions):
    if len(truth)!=len(predictions): raise ValueError('Unaligned cases')
    determinate=[(t,p) for t,p in zip(truth,predictions) if t in LABELS]
    matrix=[[sum(t==a and p==b for t,p in determinate) for b in LABELS] for a in LABELS]
    denominators={c:sum(t==c for t,p in determinate) for c in LABELS}
    completed=[(t,p) for t,p in determinate if p in LABELS]
    recalls={};scores=[]
    for c in LABELS:
        tp=sum(t==c and p==c for t,p in completed);fp=sum(t!=c and p==c for t,p in completed);fn=sum(t==c and p!=c for t,p in completed)
        recalls[c]=tp/(tp+fn) if tp+fn else None
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return {'label_order':list(LABELS),'confusion_matrix_completed':matrix,'truth_denominators':denominators,
        'completed_prediction_denominators':{c:sum(t==c for t,p in completed) for c in LABELS},
        'recall_completed':recalls,'macro_f1_completed_fixed_three_labels':sum(scores)/3 if completed else None,
        'truth_coverage':len(determinate)/len(truth) if truth else None,'truth_abstentions':len(truth)-len(determinate),
        'completed_predictions_on_determinate_truth':len(completed),'failed_predictions_all_cases':sum(p not in LABELS for p in predictions),
        'failed_predictions_on_determinate_truth':len(determinate)-len(completed),
        'all_attempt_success_rate_determinate':sum(t==p for t,p in completed)/len(determinate) if determinate else None,
        'per_class_all_attempt_success':{c:sum(t==c and p==c for t,p in determinate)/n if n else None for c,n in denominators.items()},
        'class_coverage_evaluable':all(n>=2 for n in denominators.values())}
