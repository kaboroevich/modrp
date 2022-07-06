import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SupervisedEncoder(nn.Module):

    def __init__(self, input_dim: int, output_dim: int,
                 drop_rate: float) -> None:
        super(SupervisedEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.model(x)
        return output


class Classifier(nn.Module):

    def __init__(self, input_dim: int, output_dim: int,
                 drop_rate: float) -> None:
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.model(x)
        return output


class MoliNet(nn.Module):

    def __init__(self,
                 exp_in: int, exp_out: int, exp_dr: float,
                 mut_in: int, mut_out: int, mut_dr: float,
                 cna_in: int, cna_out: int, cna_dr: float,
                 cls_dr: float) -> None:
        super(MoliNet, self).__init__()
        self.ese = SupervisedEncoder(exp_in, exp_out, exp_dr)
        self.mse = SupervisedEncoder(mut_in, mut_out, mut_dr)
        self.cse = SupervisedEncoder(cna_in, cna_out, cna_dr)
        cls_in = exp_out + mut_out + cna_out
        self.cls = Classifier(cls_in, 1, cls_dr)

    def forward(self, exp_x: Tensor, mut_x: Tensor,
                cna_x: Tensor) -> Tuple[Tensor, Tensor]:
        exp_enc = self.ese(exp_x)
        mut_enc = self.mse(mut_x)
        cna_enc = self.cse(cna_x)

        omics = torch.cat((exp_enc, mut_enc, cna_enc), 1)
        omics = F.normalize(omics, p=2, dim=0)
        output = self.cls(omics)
        return omics, output
