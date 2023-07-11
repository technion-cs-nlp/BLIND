""" Taken from https://github.com/rabeehk/robust-nli"""
import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, temperature=1.0, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.size_average = size_average

    def compute_probs(self, inputs, targets):
        prob_dist = F.softmax(inputs, dim=1)
        pt = prob_dist.gather(1, targets)
        return pt

    def compute_probs_temp(self, inputs, targets):
        prob_dist = F.softmax(inputs / self.temperature, dim=1)
        pt = prob_dist.gather(1, targets)
        return pt

    def aggregate(self, p1, p2, operation):
        if self.aggregate_ensemble == "mean":
            result = (p1+p2)/2
            return result
        elif self.aggregate_ensemble == "multiply":
            result = p1*p2
            return result
        else:
            assert NotImplementedError("Operation ", operation, "is not implemented.")

    def forward(self, inputs, targets, inputs_adv, targets_adv, second_inputs_adv=None):

        targets = targets.view(-1, 1)
        targets_adv = targets_adv.view(-1, 1)
        pt = self.compute_probs(inputs, targets)
        pt_scale = self.compute_probs_temp(inputs_adv, targets_adv)
        a = torch.pow((1 - pt_scale), self.gamma)
        batch_loss = -self.alpha * a * torch.log(pt)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss