import unittest
import torch
from torch import nn
from setsup_recal import self_recalibration, uniform_mixing, stage2_adapter, set_supervised_loss, center_relation_calibration, evaluate_distribution_batch, aggregate_volume_slice_rows
from setsup_recal.controls import CenterOnlyMixer, DispersionOnlyMixer, TwoScalarMixer


class Heads(nn.Module):
    def __init__(self):
        super().__init__(); self.linear=nn.Linear(1,6)
    def forward(self,features,candidates,index):
        return self.linear(features.mean((2,3))).unsqueeze(1) + candidates[:,index:index+1]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__(); self.linear=nn.Linear(6,1)
    def forward(self,features,latent,use_softmax=False):
        return features+self.linear(latent)[:,:,None,None]


class Core(nn.Module):
    """A synthetic interface fixture, not a substitute for D-Persona."""
    def __init__(self):
        super().__init__();self.latent_dim=6;self.num_experts=3
        self.unet=nn.Conv2d(1,1,1);self.prior=nn.Linear(1,1);self.posterior=nn.Linear(1,1)
        self.fcomb=Decoder();self.proj_heads=Heads()
    def forward(self,images,unused,training=False):
        self.unet_features=self.unet(images)
        self.prior_latent_space=torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(images.shape[0],6),torch.ones(images.shape[0],6)),1)


class MethodChecks(unittest.TestCase):
    def setUp(self): torch.manual_seed(18)

    def test_identity_and_parameter_count(self):
        for count in (3,6,7):
            module=self_recalibration(count);z=torch.randn(2,count,6)
            self.assertEqual(sum(p.numel() for p in module.parameters()),37)
            self.assertTrue(torch.equal(z,module(z)))
            self.assertTrue(torch.equal(module.projection.weight,torch.eye(6)))

    def test_no_other_rater_input_after_training(self):
        module=self_recalibration(6)
        with torch.no_grad():module.gate.fill_(.7);module.projection.weight.normal_()
        z=torch.randn(2,6,6);changed=z.clone();changed[:,1:]+=torch.randn_like(changed[:,1:])*9
        torch.testing.assert_close(module(z)[:,0],module(changed)[:,0],rtol=0,atol=0)
        perm=torch.tensor([2,5,0,3,1,4])
        torch.testing.assert_close(module(z[:,perm]),module(z)[:,perm])

    def test_shared_parameter_operator_decomposition(self):
        full=uniform_mixing(6);center=CenterOnlyMixer(6);spread=DispersionOnlyMixer(6,6)
        with torch.no_grad():
            full.gate.fill_(.4);full.projection.weight.normal_()
            for module in (center,spread):
                module.gate.copy_(full.gate);module.projection.weight.copy_(full.projection.weight)
        z=torch.randn(4,6,6)
        torch.testing.assert_close(center(z)+spread(z)-z,full(z),rtol=1e-5,atol=1e-6)
        simple=TwoScalarMixer(6)
        self.assertEqual(sum(p.numel() for p in simple.parameters()),2)
        self.assertTrue(torch.equal(simple(z),z))

    def test_set_losses_ignore_order_but_indexed_loss_does_not(self):
        p=torch.rand(2,3,5,4);t=torch.randint(0,2,p.shape).float()
        first=center_relation_calibration(p,t)
        second=center_relation_calibration(p[:,[2,0,1]],t[:,[2,0,1]])
        for k in ('center_loss','relation_loss'):torch.testing.assert_close(first[k],second[k])
        logits=torch.tensor([[[[8.,-8.]], [[-8.,8.]]]])
        targets=(logits>0).float()
        aligned=set_supervised_loss(logits,targets,mean_weight=0,pair_weight=0)['base']
        swapped=set_supervised_loss(logits[:,[1,0]],targets,mean_weight=0,pair_weight=0)['base']
        self.assertLess(float(aligned),.001);self.assertGreater(float(swapped),.99)

    def test_pair_excludes_diagonal_and_mean_uses_squared_denominator(self):
        # Opposite hard masks have zero off-diagonal IoU; identical predicted
        # soft masks have IoU 1/3. Nonzero diagonal soft-IoU errors are excluded.
        p=torch.full((1,2,1,2),.5)
        t=torch.tensor([[[[1.,0.]],[[0.,1.]]]])
        result=center_relation_calibration(p,t)
        self.assertAlmostEqual(float(result['center_loss']),0.,places=6)
        expected=((.5+1e-6)/(1.5+1e-6)-1e-6/(2+1e-6))**2
        self.assertAlmostEqual(float(result['relation_loss']),expected,places=6)

    def test_ged_analytic_extremes_and_anonymous_permutation(self):
        target=torch.tensor([[[[1.,0.]],[[0.,1.]]]])
        identical=evaluate_distribution_batch(target,target)
        self.assertAlmostEqual(float(identical['ged']),0.,places=6)
        swapped=evaluate_distribution_batch(target[:,[1,0]],target)
        for key in identical:torch.testing.assert_close(identical[key],swapped[key])
        allzero=torch.zeros(1,2,1,2);allone=torch.ones_like(allzero)
        self.assertAlmostEqual(float(evaluate_distribution_batch(allzero,allone)['ged']),2.,places=5)

    def test_npc_full_volume_differs_from_slice_average(self):
        rows=[]
        for index in range(2):
            target=torch.ones(2,2,2)
            pred=target.clone() if index==0 else torch.zeros_like(target)
            rows.append(dict(patient_id='synthetic',volume_id='synthetic',slice_index=index,rater_ids=['a','b'],generated=pred,targets=target))
        result=aggregate_volume_slice_rows(rows,aligned_personas=True)
        self.assertAlmostEqual(result['patient_macro']['consensus_dice'],2/3,places=5)
        self.assertAlmostEqual(result['sample_macro']['consensus_dice'],.5,places=5)
        self.assertIsNone(aggregate_volume_slice_rows(rows)['mean_per_rater_dice'])
        with self.assertRaises(ValueError):aggregate_volume_slice_rows([rows[1]])
        with self.assertRaises(ValueError):aggregate_volume_slice_rows(rows+[rows[0]])

    def test_gradient_and_frozen_core_contract(self):
        adapter=stage2_adapter(Core());adapter.train()
        for component in (adapter.core.unet,adapter.core.prior,adapter.core.posterior,adapter.core.fcomb):
            self.assertFalse(component.training)
            self.assertTrue(all(not p.requires_grad for p in component.parameters()))
        self.assertTrue(adapter.core.proj_heads.training)
        with torch.no_grad():adapter.relation_mixer.gate.fill_(.2)
        before={name:p.detach().clone() for name,p in adapter.named_parameters()}
        images=torch.rand(2,1,4,4)
        targets=torch.randint(0,2,(2,3,4,4)).float()
        logits=adapter.persona_logits(images,candidate_samples=100,generator=torch.Generator().manual_seed(11),use_graph=True)
        loss=set_supervised_loss(logits,targets,mean_weight=.1,pair_weight=.1)['total']
        loss.backward()
        for group in (adapter.core.proj_heads,adapter.relation_mixer):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters()))
        optimizer=torch.optim.Adam(adapter.trainable_parameters('rg_persona'),lr=.01);optimizer.step()
        for name,p in adapter.named_parameters():
            if not p.requires_grad:self.assertTrue(torch.equal(p,before[name]))
        self.assertFalse(torch.equal(adapter.relation_mixer.gate,before['relation_mixer.gate']))


if __name__=='__main__':unittest.main()
