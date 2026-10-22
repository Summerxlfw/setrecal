"""Portable entry points for the published Stage-II development protocol."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .adapter import DPersonaStage2Adapter
from .data import MultiRaterDataset, validate_disjoint_manifests
from .fingerprint import state_dict_sha256
from .stage1 import build_dpersona_stage1_adapter
from .training import _make_loader, _run_arm, evaluate_persona_loader
from .source_metrics import validate_split, evaluate_persona_source_loader


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_protocol(protocol_root, task):
    root = Path(protocol_root).resolve()
    config = json.loads((root / 'configurations.json').read_text())[task]
    provenance = json.loads((root / 'SOURCE_FINGERPRINTS.json').read_text())
    expected = {row['file']: row['portable_sha256'] for row in provenance['manifests']}
    manifests = {}
    for split in ('train', 'validation'):
        name = f'manifests/{task}_{split}.json'
        path = root / name
        if sha256(path) != expected[name]:
            raise ValueError(f'Published split manifest changed: {name}')
        value = json.loads(path.read_text())
        if value['split'] != split or value['task_id'] != config['task_id']:
            raise ValueError(f'Task/split mismatch: {name}')
        if value['test_opened'] is not False:
            raise ValueError('This entry point uses the published development protocol')
        if len(value['records']) != config[split + '_count']:
            raise ValueError(f'Sample count mismatch: {name}')
        if any(len(row['rater_ids']) != config['rater_count'] for row in value['records']):
            raise ValueError(f'Rater count mismatch: {name}')
        manifests[split] = value
    train_ids = {row['patient_id'] for row in manifests['train']['records']}
    validation_ids = {row['patient_id'] for row in manifests['validation']['records']}
    if train_ids & validation_ids:
        raise ValueError('Patient overlap between training and validation')
    if config['selection_scope'] == 'source_macro':
        groups = validate_split(manifests['train']['records'], manifests['validation']['records'])
        if len(groups) != config['validation_source_count']:
            raise ValueError('Source count differs from the published protocol')
    return config, manifests


def bind_manifest(value, cache_root, destination):
    """Bind a portable manifest to a local cache without changing its records."""
    bound = copy.deepcopy(value)
    root = Path(cache_root).resolve()
    for record in bound['records']:
        relative = Path(record['cache_file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('cache_file must be relative and stay inside the cache')
        if not (root / relative).resolve().is_relative_to(root):
            raise ValueError('Cache path resolves outside the selected cache root')
    bound['cache_root'] = str(root)
    Path(destination).write_text(json.dumps(bound, indent=2) + '\n')
    return Path(destination)


def verify_upstream(source_root, protocol_root):
    root = Path(source_root).resolve()
    upstream = json.loads((Path(protocol_root) / 'UPSTREAM.json').read_text())
    if not (root / 'LICENSE').is_file():
        raise FileNotFoundError('D-Persona checkout must include its LICENSE')
    for name, expected in upstream['files'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'D-Persona source differs from the recorded snapshot: {name}')
    return upstream['commit']


def guard_upstream_imports(source_root):
    """Bind generic upstream package names to the checked-out source tree."""
    code_root = (Path(source_root).resolve() / 'D-Persona' / 'code').resolve()
    prefixes = ('lib', 'Probabilistic_Unet_Pytorch', 'pionono_models', 'utils', 'configs')
    for name, module in list(sys.modules.items()):
        if module is None or not any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            continue
        locations = list(getattr(module, '__path__', []))
        filename = getattr(module, '__file__', None)
        if filename:
            locations.append(filename)
        if not locations or any(not Path(location).resolve().is_relative_to(code_root) for location in locations):
            raise RuntimeError(f'Upstream module {name} is already loaded from another source; use a fresh Python process')
    code_path = str(code_root)
    if code_path in sys.path:
        sys.path.remove(code_path)
    sys.path.insert(0, code_path)


def validate_checkpoint_config(checkpoint_config, config, task):
    expected = {'arm': 'rg_persona', 'task_id': config['task_id'], 'seed': config['seed'],
                'epochs': config['epochs'], 'center_weight': config['center_weight'],
                'relation_weight': config['relation_weight'], 'learning_rate': config['learning_rate'],
                'candidate_samples': config['candidate_samples'], 'output_masks': config['rater_count']}
    if any(checkpoint_config.get(key) != value for key, value in expected.items()):
        raise ValueError('Stage-II checkpoint configuration differs from the selected task')
    allowed = {f'validation.{config["selection_scope"]}.ged'}
    if config['selection_scope'] == 'source_macro':
        # The archived trainer used image_macro as its field label for source averages.
        allowed.add('validation.image_macro.ged')
    if task == 'riga_cup' and config['selection_scope'] == 'image_macro':
        allowed.add('validation.image_macro.ged_k6')  # Original cup metadata label.
    if checkpoint_config.get('selection_metric') not in allowed:
        raise ValueError('Checkpoint selection metric differs from the published protocol')
    if checkpoint_config.get('test_opened') is not False:
        raise ValueError('Checkpoint is not from the development protocol')
    hashes = checkpoint_config.get('hashes', {})
    for key, expected_hash in [('stage1_checkpoint', config['stage1_sha256']),
                               ('initial_state', config['initial_state_sha256'])]:
        if config['selection_scope'] == 'source_macro' and key not in hashes:
            raise ValueError(f'Current RIGA checkpoint requires its {key} fingerprint')
        if key in hashes and hashes[key] != expected_hash:
            raise ValueError(f'Checkpoint {key} fingerprint differs from the study')


def build_self_recalibration_model(*, source_root, stage1_checkpoint,
                                    relation_prior_path, input_channels,
                                    rater_count, device):
    # The proposed configuration replaces the relation buffer with an identity
    # matrix. No estimated relation prior is needed to reproduce that buffer.
    del relation_prior_path
    guard_upstream_imports(source_root)
    stage1 = build_dpersona_stage1_adapter(source_root=source_root,
        input_channels=input_channels, rater_count=rater_count, device=device)
    guard_upstream_imports(source_root)
    checkpoint = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
    stage1.load_state_dict(checkpoint['model_state'], strict=True)
    model = DPersonaStage2Adapter(stage1.core, torch.eye(rater_count, device=device)).to(device)
    model.control_variant = 'self_only'
    return model


def verify_initial_state(model, config):
    observed = state_dict_sha256(model)
    if observed != config['initial_state_sha256']:
        raise ValueError('Initial state differs from the recorded study state; '
                         'check Stage-I weights, D-Persona source and environment. '
                         f'Expected {config["initial_state_sha256"]}, obtained {observed}.')
    return observed


def hooks_for(task, manifests=None, config=None):
    if config and config['selection_scope'] == 'source_macro':
        from functools import partial
        groups = validate_split(manifests['train']['records'], manifests['validation']['records'])
        return dict(validation_evaluator=partial(evaluate_persona_source_loader, groups=groups),
                    selection_scope='source_macro')
    if task != 'npc':
        return {}
    from .npc_augmentation import npc170_training_augment
    from .volume_metrics import evaluate_persona_volume_loader
    return dict(validation_evaluator=evaluate_persona_volume_loader,
                selection_scope='patient_macro', batch_augmenter=npc170_training_augment)


def run(args):
    protocol = Path(args.protocol).resolve()
    config, manifests = read_protocol(protocol, args.task)
    report = {'task': args.task, 'train_samples': len(manifests['train']['records']),
              'validation_samples': len(manifests['validation']['records']),
              'raters': config['rater_count'], 'selection_scope': config['selection_scope'],
              'seed': config['seed'], 'epochs': config['epochs'],
              'stage1_sha256': config['stage1_sha256']}
    identity_unit = 'patients' if args.task == 'npc' else 'image_ids'
    for split in ('train', 'validation'):
        report[f'{split}_{identity_unit}'] = len({r['patient_id'] for r in manifests[split]['records']})
    if config['selection_scope'] == 'source_macro':
        report['validation_sources'] = config['validation_source_count']
    if args.command == 'inspect':
        return dict(report, status='METADATA_CHECKED', image_arrays_loaded=False)
    cache_root = Path(args.cache_root).resolve()
    with tempfile.TemporaryDirectory(prefix='setsup_recal_manifests_') as temporary:
        paths = [bind_manifest(manifests[split], cache_root, Path(temporary) / (split + '.json'))
                 for split in ('train', 'validation')]
        validate_disjoint_manifests(paths)
        data = [MultiRaterDataset(path) for path in paths]
        for value in manifests.values():
            for row in value['records']:
                if sha256(cache_root / row['cache_file']) != row['cache_sha256']:
                    raise ValueError(f'Cached file does not match the study: {row["cache_file"]}')
        if args.command == 'validate-cache':
            for dataset in data:
                for sample in dataset:
                    if not bool(torch.isfinite(sample['image']).all()):
                        raise ValueError('Cached image has non-finite values')
            return dict(report, status='CACHE_VALIDATED', image_arrays_loaded=True)
        verify_upstream(args.source_root, protocol)
        if sha256(args.stage1_checkpoint) != config['stage1_sha256']:
            raise ValueError('Stage-I checkpoint does not match the recorded study weights')
        device = torch.device(args.device)
        torch.set_num_threads(4)
        seed = config['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        model_kwargs = dict(source_root=Path(args.source_root),
                            stage1_checkpoint=Path(args.stage1_checkpoint),
                            relation_prior_path=None, input_channels=config['input_channels'],
                            rater_count=config['rater_count'], device=device)
        model = build_self_recalibration_model(**model_kwargs)
        initial = copy.deepcopy(model.state_dict())
        initial_hash = verify_initial_state(model, config)
        validation = _make_loader(data[1], config['batch_size'], seed + 4000, False, 0)
        hooks = hooks_for(args.task, manifests, config)
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(f'Choose a new output path: {output}')
        if args.command == 'evaluate':
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            validate_checkpoint_config(checkpoint['config'], config, args.task)
            model.load_state_dict(checkpoint['model_state'], strict=True)
            identity = torch.eye(config['rater_count'], device=device)
            if not torch.equal(model.relation_mixer.relation_prior, identity):
                raise ValueError('Checkpoint is not the self-recalibration configuration')
            evaluator = hooks.get('validation_evaluator', evaluate_persona_loader)
            result = evaluator(model, validation, arm='rg_persona', device=device,
                               seed=seed + 90_000_000, candidate_samples=config['candidate_samples'])
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x') as handle:
                json.dump(result, handle, indent=2); handle.write('\n')
            return dict(report, status='EVALUATED', output=str(output),
                        checkpoint_sha256=sha256(args.checkpoint))
        del model
        output.parent.mkdir(parents=True, exist_ok=True)
        result = _run_arm(arm='rg_persona', output_dir=output,
            model_factory=build_self_recalibration_model, source_root=model_kwargs['source_root'],
            stage1_checkpoint=model_kwargs['stage1_checkpoint'], relation_prior_path=None,
            dataset=data[0], train_loader=_make_loader(data[0], config['batch_size'], seed + 3000, True, 0),
            validation_loader=validation, initial_state=initial,
            center_weight=config['center_weight'], relation_weight=config['relation_weight'],
            seed=seed, epochs=config['epochs'], learning_rate=config['learning_rate'],
            candidate_samples=config['candidate_samples'], device=device,
            hashes={'stage1_checkpoint': config['stage1_sha256'], 'initial_state': initial_hash,
                    'train_manifest': sha256(protocol / f'manifests/{args.task}_train.json'),
                    'validation_manifest': sha256(protocol / f'manifests/{args.task}_validation.json')},
            **hooks)
        return dict(report, status='TRAINED', output=str(output), summary=result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('inspect', 'validate-cache', 'train', 'evaluate'):
        action = sub.add_parser(command)
        action.add_argument('--task', choices=('riga_cup', 'riga_disc', 'refuge', 'npc'), required=True)
        action.add_argument('--protocol', type=Path, default=Path('protocol'))
        if command != 'inspect':
            action.add_argument('--cache-root', type=Path, required=True,
                                help='Directory containing the manifest cache_file paths')
        if command in ('train', 'evaluate'):
            action.add_argument('--source-root', type=Path, required=True)
            action.add_argument('--stage1-checkpoint', type=Path, required=True)
            action.add_argument('--device', default='cpu', help='cpu or an explicit CUDA device')
            action.add_argument('--output', type=Path, required=True)
        if command == 'evaluate':
            action.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
