"""Require source-disjoint development and reduce duplicate views per source."""
import math
from collections import defaultdict
from statistics import fmean

def validate_split(train, validation):
    for label, rows in [('train', train), ('validation', validation)]:
        ids = [r['sample_id'] for r in rows]
        if not rows or len(ids) != len(set(ids)):
            raise ValueError(label + ': empty or repeated sample IDs')
    train_sources = {r['source_sha256']['image'] for r in train}
    validation_sources = {r['source_sha256']['image'] for r in validation}
    if train_sources & validation_sources:
        raise ValueError('source image crosses train/development')
    groups = defaultdict(list)
    for r in validation:
        groups[r['source_sha256']['image']].append(r['sample_id'])
    return {k: sorted(v) for k, v in sorted(groups.items())}

def source_macro(rows, groups, metrics):
    """Keep all labels/crops; average within source then equally across sources."""
    by_id = {r['sample_id']: r for r in rows}
    expected = [i for ids in groups.values() for i in ids]
    if len(expected) != len(set(expected)) or any(not ids for ids in groups.values()):
        raise ValueError('invalid source grouping')
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ValueError('missing, extra or repeated evaluation IDs')
    result = {}
    for metric in metrics:
        values = {i: float(r[metric]) for i, r in by_id.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('nonfinite metric: ' + metric)
        result[metric] = fmean(fmean(values[i] for i in ids) for ids in groups.values())
    return result


def evaluate_persona_source_loader(model, loader, *, groups, **kwargs):
    """Compute unchanged per-view metrics, then average equally over sources."""
    from .training import evaluate_persona_loader
    result = evaluate_persona_loader(model, loader, **kwargs)
    rows = [dict(row) for row in result['rows']]
    rater_count = len(result['per_rater_dice'])
    for row in rows:
        if len(row['per_rater_dice']) != rater_count:
            raise ValueError('inconsistent rater counts')
        for index, value in enumerate(row['per_rater_dice']):
            row[f'rater_{index}'] = value
    metric_keys = list(result['image_macro'])
    macro = source_macro(rows, groups, metric_keys + [f'rater_{i}' for i in range(rater_count)])
    result['source_macro'] = {key: macro[key] for key in metric_keys}
    result['per_rater_dice'] = [macro[f'rater_{i}'] for i in range(rater_count)]
    result['mean_per_rater_dice'] = sum(result['per_rater_dice']) / rater_count
    result['evaluation_unit'] = 'source_image_macro'
    result['source_count'] = len(groups)
    result['validation_views'] = len(rows)
    return result
