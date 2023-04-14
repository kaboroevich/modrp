import torch.optim as optim
from .nn import SuperFeltNet


class SuperFeltAdagrad(optim.Adagrad):

    def __init__(self, net: SuperFeltNet,
                 enc_lr: float, enc_wd: float,
                 cls_lr: float, cls_wd: float):
        params = [{'params': net.ese.parameters(),
                   'lr': enc_lr, 'weight_decay': enc_wd},
                  {'params': net.mse.parameters(),
                   'lr': enc_lr, 'weight_decay': enc_wd},
                  {'params': net.cse.parameters(),
                   'lr': enc_lr, 'weight_decay': enc_wd},
                  {'params': net.cls.parameters(),
                   'lr': cls_lr, 'weight_decay': cls_wd}]
        super(SuperFeltAdagrad, self).__init__(params)
