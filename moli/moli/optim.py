import torch.optim as optim
from .nn import MoliNet


class MoliAdagrad(optim.Adagrad):

    def __init__(self, net: MoliNet,
                 exp_lr: float, mut_lr: float, cna_lr: float,
                 cls_lr: float, cls_wd: float):
        params = [{'params': net.ese.parameters(), 'lr': exp_lr},
                  {'params': net.mse.parameters(), 'lr': mut_lr},
                  {'params': net.cse.parameters(), 'lr': cna_lr},
                  {'params': net.cls.parameters(),
                   'lr': cls_lr, 'weight_decay': cls_wd}]
        super(MoliAdagrad, self).__init__(params)
