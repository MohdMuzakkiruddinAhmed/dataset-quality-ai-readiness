"""Task-schema missingness adapters, blinded repair and natural-aware evaluation."""

import numpy as np
import pandas as pd

from dataset_readiness.artifacts import csv_bytes, json_bytes, sha256
from dataset_readiness.baselines.rules import assess_rules
from dataset_readiness.corruption.missing import inject_missing_values
from dataset_readiness.corruption.schemas import MissingnessConfig
from dataset_readiness.profiling import AssessmentSchema, ProfileOptions, build_profile
from dataset_readiness.remediation.core import CATEGORICAL, NUMERIC, Finding, build_plan, validate_plan

VERSION = 'task_missingness_v1'
REPAIR = 'task_missing_repair_v1'


def schema_for(task):
    return AssessmentSchema(task.target, task.numeric, task.categorical)


def inject(frame, task, severity, seed):
    task.validate_frame(frame, training=True)
    result=inject_missing_values(frame[list(task.features)],frame[task.target],MissingnessConfig(severity,seed,task.features))
    observed=result.dataframe.copy();observed[task.target]=result.labels
    task.validate_frame(observed,training=True)
    cells={c:{'rows':[],'original_values':[]} for c in task.features}
    for cell in result.manifest.affected_cells:
        cells[cell.column]['rows'].append(cell.row_id);cells[cell.column]['original_values'].append(cell.original_value)
    truth={'version':VERSION,'engine_version':result.manifest.engine_version,'severity':severity,'seed':seed,
           'selected_columns':list(task.features),'eligible_cells':result.manifest.eligible_cell_count,
           'natural_missing_cells':result.manifest.preexisting_missing_count,'injected_cells':len(result.manifest.affected_cells),
           'cells_by_column':cells,'reference_sha256':sha256(csv_bytes(frame)),'observed_sha256':sha256(csv_bytes(observed)),
           'selection':'Decimal floor(severity * observed feature cells); PCG64(seed), row-major candidates, choice without replacement, sorted selected offsets',
           'scope':'Evaluator only; no natural-missing original values are known'}
    return observed,truth


def observe(frame, task):
    """Only observed training data and declared metadata enter this boundary."""
    task.validate_frame(frame,training=True)
    schema=schema_for(task)
    return {'protocol':VERSION,'input_sha256':sha256(csv_bytes(frame)),
            'profile':build_profile(frame,schema,ProfileOptions()),'rules':assess_rules(frame,schema).to_dict(),
            'readiness':None,'scope':'Observed missingness and other descriptive flags; missingness origin is unknown to rules'}


def findings_from(report, task):
    return tuple(Finding(i['issue_id'],i['issue_type'],i['level'],i['column'],tuple(i['rows']),
                         NUMERIC if i['column'] in task.numeric else CATEGORICAL)
                 for i in report['rules']['issues'] if i['issue_type']=='missing_values' and i['level']=='cell')


def equal_cells(left,right,features):
    return np.column_stack([(left[c].isna() & right[c].isna() | left[c].eq(right[c]).fillna(False)).to_numpy(dtype=bool) for c in features])


def verify(before,after,task,report,plan,manifest):
    task.validate_frame(before,training=True);task.validate_frame(after,training=True)
    if (not before.index.equals(after.index) or not before.index.dtype==after.index.dtype
            or not before.dtypes.equals(after.dtypes) or not before[task.target].equals(after[task.target])):
        raise ValueError('Repair changed protected rows, labels or dtypes')
    if report['input_sha256']!=sha256(csv_bytes(before)):
        raise ValueError('Assessment input fingerprint mismatch')
    validate_plan(plan,before,schema_for(task),findings_from(report,task))
    expected=np.zeros((len(before),len(task.features)),dtype=bool)
    changes=[]
    for step in plan.steps:
        rows=before.index.get_indexer(step.rows);col=task.features.index(step.column)
        if (rows<0).any() or not before.loc[list(step.rows),step.column].isna().all():raise ValueError('Invalid repair coordinates')
        expected[rows,col]=True
        if not after.loc[list(step.rows),step.column].eq(step.fill_value).all():raise ValueError('Wrong repair fill')
        changes.append({'column':step.column,'rows':list(step.rows),'before':None,'after':step.fill_value,
                        'step_id':step.step_id,'source_issue_ids':list(step.source_issue_ids)})
    actual=~equal_cells(before,after,task.features)
    if not np.array_equal(actual,expected):raise ValueError('Repair changed unplanned cells or omitted planned edits')
    exact={'version':REPAIR,'format':'column groups; after value and source IDs apply to every listed row',
           'input_sha256':sha256(csv_bytes(before)),'repaired_sha256':sha256(csv_bytes(after)),
           'plan_sha256':sha256(json_bytes(plan.to_dict())),'changed_cells':int(expected.sum()),'changes':changes}
    if json_bytes(manifest)!=json_bytes(exact):raise ValueError('Repair manifest mismatch')
    return {'status':'passed','version':REPAIR,'only_planned_missing_cells_changed':True,'rows_labels_dtypes_preserved':True,
            'changed_cells':int(actual.sum()),'remaining_missing_cells':int(after[list(task.features)].isna().sum().sum())}


def repair(frame,task,report):
    """Fit from observation alone; compact manifests retain every edited coordinate."""
    task.validate_frame(frame,training=True)
    if report['input_sha256']!=sha256(csv_bytes(frame)):raise ValueError('Assessment input fingerprint mismatch')
    plan=build_plan(frame,schema_for(task),findings_from(report,task))
    after=frame.copy(deep=True);changes=[]
    for step in plan.steps:
        after.loc[list(step.rows),step.column]=step.fill_value
        changes.append({'column':step.column,'rows':list(step.rows),'before':None,'after':step.fill_value,
                        'step_id':step.step_id,'source_issue_ids':list(step.source_issue_ids)})
    manifest={'version':REPAIR,'format':'column groups; after value and source IDs apply to every listed row',
              'input_sha256':sha256(csv_bytes(frame)),'repaired_sha256':sha256(csv_bytes(after)),
              'plan_sha256':sha256(json_bytes(plan.to_dict())),'changed_cells':sum(len(s.rows) for s in plan.steps),'changes':changes}
    verification=verify(frame,after,task,report,plan,manifest)
    post=observe(after,task)
    verification['post_repair_profile']=post['profile']
    verification['post_repair_rules']=post['rules']
    return after,plan.to_dict(),manifest,verification


def evaluate(reference,observed,repaired,task,truth,report):
    """Evaluator-only truth accounting. Natural missing fills have no restoration score."""
    for frame in (reference,observed,repaired):task.validate_frame(frame,training=True)
    if any(not reference.index.equals(frame.index) or not reference[task.target].equals(frame[task.target]) for frame in (observed,repaired)):
        raise ValueError('Evaluator row/target protection failed')
    if truth['reference_sha256']!=sha256(csv_bytes(reference)) or truth['observed_sha256']!=sha256(csv_bytes(observed)):
        raise ValueError('Evaluator truth fingerprint mismatch')
    if report['input_sha256']!=truth['observed_sha256']:raise ValueError('Evaluator assessment fingerprint mismatch')
    features=list(task.features);natural=reference[features].isna().to_numpy();missing=observed[features].isna().to_numpy()
    injected=np.zeros_like(natural);detected=np.zeros_like(natural)
    for col,entries in truth['cells_by_column'].items():
        if col not in features or len(entries['rows'])!=len(entries['original_values']) or len(set(entries['rows']))!=len(entries['rows']):
            raise ValueError('Invalid injected truth coordinates')
        rows=reference.index.get_indexer(entries['rows']);column=features.index(col)
        if (rows<0).any():raise ValueError('Unknown truth row')
        injected[rows,column]=True
        if not reference.loc[entries['rows'],col].reset_index(drop=True).eq(pd.Series(entries['original_values'],dtype=object)).all():
            raise ValueError('Injected original values mismatch')
    if injected.sum()!=truth['injected_cells'] or (injected&natural).any() or not np.array_equal(missing,natural|injected):
        raise ValueError('Natural/injected missingness partition mismatch')
    if (~equal_cells(reference,observed,features)&~injected).any():raise ValueError('Corruption changed protected cells')
    for issue in report['rules']['issues']:
        if issue['issue_type']!='missing_values' or issue['level']!='cell':continue
        if issue['column'] not in features:raise ValueError('Invalid detection column')
        rows=reference.index.get_indexer(issue['rows'])
        if (rows<0).any():raise ValueError('Invalid detection row')
        detected[rows,features.index(issue['column'])]=True
    changed=~equal_cells(observed,repaired,features)
    if (changed&~missing).any():raise ValueError('Repair changed observed protected cells')
    restored=equal_cells(reference,repaired,features)&injected
    remaining=repaired[features].isna().to_numpy()
    ratio=lambda n,d:float(n/d) if d else None
    count=lambda mask:int(mask.sum())
    return {'version':VERSION,'detection':{
        'all_missing_findings':count(detected),'observed_missing_cells':count(missing),
        'observed_missing_precision':ratio(count(detected&missing),count(detected)),
        'observed_missing_recall':ratio(count(detected&missing),count(missing)),
        'injected_cells':count(injected),'injected_detected':count(detected&injected),'injected_recall':ratio(count(detected&injected),count(injected)),
        'injected_precision_on_originally_observed_cells':ratio(count(detected&injected),count(detected&~natural)),
        'natural_missing_cells':count(natural),'natural_missing_findings':count(detected&natural),
        'injected_share_of_all_missing_findings':ratio(count(detected&injected),count(detected)),
        'findings_on_nonmissing_cells':count(detected&~missing),
        'interpretation':'Natural findings are real observed missingness, not recovered synthetic truth; injected precision excludes naturally missing coordinates.'},
        'repair':{'changed_cells':count(changed),'injected_filled':count(changed&injected),'injected_exactly_restored':count(restored),
                  'injected_exact_restoration_rate':ratio(count(restored),count(injected)),
                  'injected_remaining_missing':count(injected&remaining),'natural_filled':count(natural&changed),
                  'natural_remaining_missing':count(natural&remaining),'natural_original_value_restoration':None,
                  'protected_observed_cell_changes':count(changed&~missing),'rows_labels_preserved':True,
                  'interpretation':'Restoration denominator includes injected cells only; original values of naturally missing cells are unknown.'},
        'readiness':None}


def metric_effects(reference,reference_repaired,observed,repaired):
    result={}
    for metric in ('macro_f1','roc_auc','average_precision'):
        base,natural,bad,fixed=(values[metric] for values in (reference,reference_repaired,observed,repaired))
        loss=base-bad
        result[metric]={'reference':base,'reference_repaired':natural,'corrupted':bad,'repaired':fixed,
            'natural_repair_delta':natural-base,'corruption_loss':loss,'repair_delta':fixed-bad,
            'repaired_minus_reference':fixed-base,'repaired_minus_reference_repaired':fixed-natural,
            'recovery_ratio':(fixed-bad)/loss if loss>0 else None,
            'recovery_reason':None if loss>0 else 'no_positive_corruption_loss',
            'repair_direction':'improved' if fixed>bad else 'harmed' if fixed<bad else 'neutral'}
    return result
