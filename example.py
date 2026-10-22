"""CPU software demonstration using synthetic values, not research data."""
import torch
from setsup_recal import self_recalibration, set_supervised_loss, evaluate_distribution_batch

torch.manual_seed(7)
module=self_recalibration(6)
z=torch.randn(2,6,6)
assert torch.equal(z,module(z))
logits=torch.randn(2,6,16,16,requires_grad=True)
targets=torch.randint(0,2,logits.shape).float()
losses=set_supervised_loss(logits,targets,mean_weight=.1,pair_weight=.1)
losses['total'].backward()
metrics=evaluate_distribution_batch(logits.detach().sigmoid(),targets)
print('Synthetic software example passed; added parameters:',sum(p.numel() for p in module.parameters()))
print('Loss fields:',sorted(losses),'Metric fields:',sorted(metrics))
