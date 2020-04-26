'''

The comments are created by Xiaoyu Zhu at 26 April.
*This build.py code has been modified by Xiaoyu Zhu for TDT4265 final project.

*Additional Support:
1. Added the support for different optimizer.
   

'''

import torch
from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model, lr=None):
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    if cfg.SOLVER.CHOICE == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.CHOICE == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


def make_lr_scheduler(cfg, optimizer, milestones=None):
    return WarmupMultiStepLR(optimizer=optimizer,
                             milestones=cfg.SOLVER.LR_STEPS if milestones is None else milestones,
                             gamma=cfg.SOLVER.GAMMA,
                             warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                             warmup_iters=cfg.SOLVER.WARMUP_ITERS)
