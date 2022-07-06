import torch.optim as optim
from .nn import SuperFeltNet


class SuperFeltAdagrad(optim.Adagrad):

    def __init__(self, net: SuperFeltNet,
                 exp_lr: float, exp_wd: float,
                 mut_lr: float, mut_wd: float,
                 cna_lr: float, cna_wd: float,
                 cls_lr: float, cls_wd: float):
        params = [{'params': net.ese.parameters(),
                   'lr': exp_lr, 'weight_decay': exp_wd},
                  {'params': net.mse.parameters(),
                   'lr': mut_lr, 'weight_decay': mut_wd},
                  {'params': net.cse.parameters(),
                   'lr': cna_lr, 'weight_decay': cna_wd},
                  {'params': net.cls.parameters(),
                   'lr': cls_lr, 'weight_decay': cls_wd}]
        super(SuperFeltAdagrad, self).__init__(params)
