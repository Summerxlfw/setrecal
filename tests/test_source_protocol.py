"""RIGA source grouping and checkpoint selection checks on synthetic inputs."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from setsup_recal.source_metrics import source_macro, validate_split, evaluate_persona_source_loader
from setsup_recal.reproduce import read_protocol, hooks_for, validate_checkpoint_config
from setsup_recal.training import _run_arm, _make_loader
import test_reproduction as reproduction_fixture
from test_method import Core
from setsup_recal import stage2_adapter
ROOT = Path(__file__).resolve().parents[1]

class SourceProtocolChecks(unittest.TestCase):
    def test_equal_weight_per_source_with_unequal_view_counts(self):
        rows = [dict(sample_id='a', ged=0.), dict(sample_id='b', ged=0.), dict(sample_id='c', ged=1.)]
        self.assertEqual(source_macro(rows, {'s1': ['a', 'b'], 's2': ['c']}, ['ged']), {'ged': .5})
        self.assertNotEqual(.5, sum(r['ged'] for r in rows)/3)

    def test_invalid_evaluation_rows_rejected(self):
        for rows in [[], [dict(sample_id='a', ged=0.)]*2,
                     [dict(sample_id='a', ged=float('nan'))], [dict(sample_id='b', ged=0.)]]:
            with self.assertRaises(ValueError):
                source_macro(rows, {'s': ['a']}, ['ged'])

    def test_source_overlap_and_duplicate_ids_rejected(self):
        train = [dict(sample_id='train', source_sha256={'image': 'same'})]
        dev = [dict(sample_id='dev', source_sha256={'image': 'same'})]
        with self.assertRaises(ValueError): validate_split(train, dev)
        dev[0]['source_sha256']['image'] = 'different'
        self.assertEqual(validate_split(train, dev), {'different': ['dev']})
        with self.assertRaises(ValueError): validate_split(train, dev*2)

    def test_current_and_initial_protocols_remain_distinct(self):
        for task in ['riga_cup', 'riga_disc']:
            current, manifests = read_protocol(ROOT/'protocol', task)
            legacy, initial = read_protocol(ROOT/'protocol_initial', task)
            self.assertEqual((len(manifests['train']['records']), len(manifests['validation']['records'])), (524, 127))
            self.assertEqual(len(validate_split(manifests['train']['records'], manifests['validation']['records'])), 126)
            self.assertEqual(current['selection_scope'], 'source_macro')
            self.assertEqual(legacy['validation_count'], 131)
            self.assertEqual(legacy['selection_scope'], 'image_macro')
            self.assertNotEqual(current['stage1_sha256'], legacy['stage1_sha256'])
            self.assertEqual(hooks_for(task, manifests, current)['selection_scope'], 'source_macro')
            self.assertEqual(hooks_for(task, initial, legacy), {})
        for task in ['refuge', 'npc']:
            self.assertEqual(read_protocol(ROOT/'protocol', task), read_protocol(ROOT/'protocol_initial', task))

    def test_indexed_dice_uses_same_source_weights(self):
        rows = [dict(sample_id=s, ged=v, per_rater_dice=[v, 1-v]) for s,v in [('a',0.), ('b',0.), ('c',1.)]]
        result = dict(rows=rows, image_macro={'ged': 1/3}, per_rater_dice=[1/3,2/3], mean_per_rater_dice=.5)
        with patch('setsup_recal.training.evaluate_persona_loader', return_value=copy.deepcopy(result)):
            actual = evaluate_persona_source_loader(None, None, groups={'s1':['a','b'], 's2':['c']})
        self.assertEqual(actual['source_macro']['ged'], .5)
        self.assertEqual(actual['per_rater_dice'], [.5,.5])
        self.assertEqual(actual['image_macro']['ged'], 1/3)
        self.assertEqual(actual['rows'], rows)

    def test_current_checkpoint_requires_weight_binding(self):
        config, _ = read_protocol(ROOT/'protocol', 'riga_cup')
        checkpoint = {k:config[k] for k in ['task_id','seed','epochs','center_weight','relation_weight','learning_rate','candidate_samples']}
        checkpoint.update(arm='rg_persona', output_masks=6, selection_metric='validation.source_macro.ged', test_opened=False,
                          hashes={'stage1_checkpoint':config['stage1_sha256'], 'initial_state':config['initial_state_sha256']})
        validate_checkpoint_config(checkpoint, config, 'riga_cup')
        checkpoint['selection_metric'] = 'validation.image_macro.ged'
        validate_checkpoint_config(checkpoint, config, 'riga_cup')
        checkpoint['hashes']['stage1_checkpoint'] = '0'*64
        with self.assertRaises(ValueError): validate_checkpoint_config(checkpoint, config, 'riga_cup')
        del checkpoint['hashes']['stage1_checkpoint']
        with self.assertRaises(ValueError): validate_checkpoint_config(checkpoint, config, 'riga_cup')

    def test_training_selects_source_ged_and_keeps_earliest_tie(self):
        fixture = reproduction_fixture.ReproductionChecks(); fixture.setUp()
        try:
            _, train = fixture.data('train'); _, dev = fixture.data('validation')
            initial = stage2_adapter(Core()).state_dict(); calls = []
            def evaluator(*args, **kwargs):
                index = len(calls); calls.append(index)
                return dict(source_macro={'ged':[.5,.3,.3][index]}, image_macro={'ged':[.1,.7,.01][index]},
                            per_rater_dice=[.5,.5,.5], mean_per_rater_dice=.5, rows=[{'sample_id':'validation_0'}])
            out = fixture.root/'source_run'
            result = _run_arm(arm='rg_persona', output_dir=out, model_factory=lambda **kw:stage2_adapter(Core()),
                source_root=None, stage1_checkpoint=None, relation_prior_path=None, dataset=train,
                train_loader=_make_loader(train,2,3041,True,0), validation_loader=_make_loader(dev,2,4041,False,0),
                initial_state=copy.deepcopy(initial), center_weight=.1, relation_weight=.1, seed=41, epochs=3,
                learning_rate=.001, candidate_samples=3, device=torch.device('cpu'), hashes={},
                validation_evaluator=evaluator, selection_scope='source_macro')
            self.assertEqual(result['best_epoch'], 2)
            self.assertEqual(result['best_validation_source_macro'], {'ged':.3})
            self.assertTrue((out/'best_validation_source_metrics.json').is_file())
            self.assertTrue((out/'best_validation_per_view.jsonl').is_file())
            checkpoint = torch.load(out/'best.pt', weights_only=False)
            self.assertEqual(checkpoint['config']['selection_metric'], 'validation.source_macro.ged')
        finally: fixture.tearDown()
