"""Software checks on synthetic arrays; no clinical/model-training evidence."""
import copy
import json
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from setsup_recal import stage2_adapter
from setsup_recal.data import MultiRaterDataset,validate_disjoint_manifests
from setsup_recal.reproduce import bind_manifest,verify_initial_state,hooks_for,guard_upstream_imports,validate_checkpoint_config
from setsup_recal.training import _make_loader,_run_arm,evaluate_persona_loader
from test_method import Core


class ReproductionChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        torch.set_num_threads(1);torch.manual_seed(41)
    def tearDown(self):self.tmp.cleanup()

    def data(self,split,volumes=False):
        records=[]
        import hashlib
        for index in range(2):
            path=self.root/f'{split}_{index}.npz'
            np.savez(path,image=np.full((1,4,4),.2+index*.3,dtype=np.float32),
                     masks=np.full((3,4,4),index,dtype=np.uint8))
            row=dict(patient_id=f'{split}_patient' if volumes else f'{split}_{index}',
                     sample_id=f'{split}_{index}',split=split,task_id='synthetic',
                     cache_file=path.name,image_shape=[1,4,4],mask_shape=[3,4,4],
                     rater_ids=['a','b','c'],spacing=[1.,1.,1.] if volumes else None,
                     source_sha256={'synthetic':'0'*64},cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if volumes:row.update(volume_id=f'{split}_volume',slice_index=index)
            records.append(row)
        value=dict(schema='multirater_seg_manifest/1',dataset_id='synthetic',
                   dataset_version='test',license_id='synthetic-test',task_id='synthetic',
                   modality='synthetic',split=split,image_channels=1,cache_root='unused',
                   test_opened=False,transform_receipt_sha256='0'*64,records=records)
        path=bind_manifest(value,self.root,self.root/(split+'.json'))
        return value,MultiRaterDataset(path)

    def test_portable_loader_and_split_identity(self):
        original,train=self.data('train',volumes=True);_,dev=self.data('validation',volumes=True)
        validate_disjoint_manifests([train.manifest_path,dev.manifest_path])
        self.assertEqual(train.records,original['records'])
        batch=next(iter(_make_loader(train,2,9,False,0)))
        self.assertEqual(batch['slice_index'],[0,1])
        self.assertEqual(batch['rater_ids'],[['a','b','c'],['a','b','c']])
        self.assertEqual(tuple(batch['masks'].shape),(2,3,4,4))
        overlap=json.loads(dev.manifest_path.read_text())
        for row in overlap['records']:row['patient_id']='train_patient'
        dev.manifest_path.write_text(json.dumps(overlap))
        with self.assertRaises(ValueError):validate_disjoint_manifests([train.manifest_path,dev.manifest_path])

    def test_cache_path_and_initial_state_mismatch_rejected(self):
        original,_=self.data('train')
        original['records'][0]['cache_file']='../outside.npz'
        with self.assertRaises(ValueError):bind_manifest(original,self.root,self.root/'bad.json')
        model=stage2_adapter(Core())
        with self.assertRaisesRegex(ValueError,'Initial state differs'):
            verify_initial_state(model,{'initial_state_sha256':'0'*64})

    def test_training_checkpoint_evaluation_roundtrip(self):
        _,train=self.data('train');_,dev=self.data('validation')
        initial=stage2_adapter(Core()).state_dict()
        def factory(**unused):return stage2_adapter(Core())
        summary=_run_arm(arm='rg_persona',output_dir=self.root/'run',model_factory=factory,
            source_root=None,stage1_checkpoint=None,relation_prior_path=None,dataset=train,
            train_loader=_make_loader(train,2,3041,True,0),validation_loader=_make_loader(dev,2,4041,False,0),
            initial_state=copy.deepcopy(initial),center_weight=.1,relation_weight=.1,
            seed=41,epochs=2,learning_rate=.001,candidate_samples=3,device=torch.device('cpu'),hashes={})
        self.assertEqual(summary['optimizer_steps'],2)
        checkpoint=torch.load(self.root/'run/best.pt',weights_only=False)
        model=factory();model.load_state_dict(checkpoint['model_state'])
        result=evaluate_persona_loader(model,_make_loader(dev,2,4041,False,0),arm='rg_persona',
            device=torch.device('cpu'),seed=90000041,candidate_samples=3)
        self.assertEqual(result['image_macro'],summary['best_validation_image_macro'])
        self.assertTrue(torch.equal(model.relation_mixer.relation_prior,torch.eye(3)))
        for name,value in model.state_dict().items():
            if name.startswith(('core.unet.','core.prior.','core.posterior.','core.fcomb.')):
                self.assertTrue(torch.equal(value,initial[name]),name)
        with self.assertRaises(FileExistsError):(self.root/'run').mkdir()

    def test_npc_hook_and_augmentation_preserve_units(self):
        hooks=hooks_for('npc');self.assertEqual(hooks['selection_scope'],'patient_macro')
        _,dataset=self.data('validation',volumes=True)
        images=torch.rand(2,1,4,4);masks=torch.randint(0,2,(2,3,4,4)).float()
        first=hooks['batch_augmenter'](images,masks,24)
        second=hooks['batch_augmenter'](images,masks,24)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(first,second)))
        self.assertTrue(set(torch.unique(first[1]).tolist()).issubset({0.,1.}))
        result=hooks['validation_evaluator'](stage2_adapter(Core()),_make_loader(dataset,2,0,False,0),
            arm='rg_persona',device='cpu',seed=9,candidate_samples=3)
        self.assertEqual(len(result['rows']),1)
        self.assertIn('patient_macro',result)
        self.assertEqual(hooks_for('riga_cup'),{})

    def test_upstream_import_guard_rejects_cached_foreign_module(self):
        module=types.ModuleType('lib');module.__file__=str(self.root/'foreign/lib.py')
        with patch.dict(sys.modules,{'lib':module}):
            with self.assertRaisesRegex(RuntimeError,'another source'):
                guard_upstream_imports(self.root/'official')
        module.__file__=str(self.root/'official/D-Persona/code/lib/__init__.py')
        with patch.dict(sys.modules,{'lib':module}),patch.object(sys,'path',list(sys.path)):
            guard_upstream_imports(self.root/'official')
            self.assertEqual(Path(sys.path[0]),(self.root/'official/D-Persona/code').resolve())

    def test_checkpoint_protocol_rejects_mismatched_fields(self):
        config=dict(task_id='npc_gtvp',seed=41,epochs=100,center_weight=.1,
                    relation_weight=.2,learning_rate=.0001,candidate_samples=100,
                    rater_count=4,selection_scope='patient_macro',stage1_sha256='a'*64,
                    initial_state_sha256='b'*64)
        checkpoint={key:config[key] for key in ['task_id','seed','epochs','center_weight',
                    'relation_weight','learning_rate','candidate_samples']}
        checkpoint.update(arm='rg_persona',output_masks=4,selection_metric='validation.patient_macro.ged',
                          test_opened=False,hashes={'stage1_checkpoint':'a'*64,'initial_state':'b'*64})
        validate_checkpoint_config(checkpoint,config,'npc')
        for key in checkpoint:
            invalid=copy.deepcopy(checkpoint)
            invalid[key]={'stage1_checkpoint':'c'*64} if key=='hashes' else 'incorrect'
            with self.subTest(field=key),self.assertRaises(ValueError):
                validate_checkpoint_config(invalid,config,'npc')


if __name__=='__main__':unittest.main()
