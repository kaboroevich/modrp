import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from typing import Callable


def moli_batch_triplets(target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    triplist = []
    for label in torch.unique(target):
        label_indices = torch.where(target == label)[0]
        if label_indices.shape[0] < 2:
            continue
        negative_indices = torch.where(target != label)[0]
        num_negative = negative_indices.shape[0]
        # All non-reciprical anchor-positive pairs
        anchor_positives = torch.combinations(label_indices, r=2,
                                              with_replacement=False)
        num_anchor = anchor_positives.shape[0]
        # Add all negatives for all positive pairs
        label_triplet = torch.hstack([
            anchor_positives.repeat_interleave(num_negative, dim=0),
            negative_indices[:, None].repeat(num_anchor, 1)])
        triplist.append(label_triplet)
    triplet = torch.vstack(triplist)
    return triplet[:, 0], triplet[:, 1], triplet[:, 2]


def moli_triplet_score(target: Tensor, embedding: Tensor,
                       margin: float) -> Tensor:
    anchor, positive, negative = moli_batch_triplets(target)
    loss = F.triplet_margin_loss(
        embedding[anchor, :], embedding[positive, :],
        embedding[negative, :], margin=margin)
    return loss


def moli_combination_loss(input: Tensor, target: Tensor,
                          embedding: Tensor, margin: float,
                          gamma: float) -> Tensor:
    bce_loss = F.binary_cross_entropy(input, target.view(-1, 1))
    triplet_loss = moli_triplet_score(target, embedding, margin)
    loss = bce_loss + (triplet_loss * gamma)
    return loss


class CombinationLoss(Module):

    def __init__(self, loss1: Callable, loss2: Callable,
                 gamma: float = 1.) -> None:
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.gamma = gamma

    def forward(self, input1: dict[str, Tensor],
                input2: dict[str, Tensor], target: Tensor) -> Tensor:
        loss1 = self.loss1(**input1, target=target)
        loss2 = self.loss2(**input2, target=target)
        loss = loss1 + loss2 * self.gamma
        return loss
