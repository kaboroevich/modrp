import moli
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Tuple


class SuperFeltNet(nn.Module):

    def __init__(self,
                 exp_in: int, exp_out: int, exp_ep: int,
                 mut_in: int, mut_out: int, mut_ep: int,
                 cna_in: int, cna_out: int, cna_ep: int,
                 enc_dr: float, cls_dr: float) -> None:
        super(SuperFeltNet, self).__init__()
        self.epoch = 0
        self.exp_ep = exp_ep
        self.ese = moli.nn.SupervisedEncoder(exp_in, exp_out, enc_dr)
        self.mut_ep = mut_ep
        self.mse = moli.nn.SupervisedEncoder(mut_in, mut_out, enc_dr)
        self.cna_ep = cna_ep
        self.cse = moli.nn.SupervisedEncoder(cna_in, cna_out, enc_dr)
        cls_in = exp_out + mut_out + cna_out
        self.cls = moli.nn.Classifier(cls_in, 1, cls_dr)
        self.update_grad(self.cls, False)

    def forward(self, exp_x: Tensor, mut_x: Tensor,
                cna_x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # expression
        if (self.exp_ep != -1) and (self.epoch >= self.exp_ep):
            self.update_grad(self.ese, False)
            self.exp_ep = -1
        exp_enc = self.ese(exp_x)

        # mutation
        if (self.mut_ep != -1) and (self.epoch >= self.mut_ep):
            self.update_grad(self.mse, False)
            self.mut_ep = -1
        mut_enc = self.mse(mut_x)

        # cna
        if (self.cna_ep != -1) and (self.epoch >= self.cna_ep):
            self.update_grad(self.cse, False)
            self.cna_ep = -1
        cna_enc = self.cse(cna_x)
        # combine encoders
        omics = torch.cat((exp_enc, mut_enc, cna_enc), 1)
        omics = F.normalize(omics, p=2, dim=0)

        # classifier
        if (self.exp_ep == -1) and (self.mut_ep == -1) and (self.cna_ep == -1):
            self.update_grad(self.cls, True)
        output = self.cls(omics)
        self.epoch += 1
        return exp_enc, mut_enc, cna_enc, output

    @staticmethod
    def update_grad(module: nn.Module, requires_grad: bool) -> None:
        for name, param in module.named_parameters():
            param.requires_grad = requires_grad
