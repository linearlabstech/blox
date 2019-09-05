

import torch
from torch import nn
class MaskedLoss(nn.Module):

    def forward(self,inputs,targets,mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inputs, 1, targets.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()
