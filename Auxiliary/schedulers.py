import torch
import math


def get_scheduler(optimizer, **options):

    if options['scheduler'] == 'step':        # 每经过 step_size 个 epoch，更新一次 lr，乘以 gamma
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=150)

    elif options['scheduler'] == 'plateau':
        # 发现 Loss 不再降低后（mode='min'），或者 acc 不再提升后（mode='max'），降低 lr
        # patience：不再减小或者增大的累计次数
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                               patience=int(options['max_epoch'] / 10))

    elif options['scheduler'] == 'cosine':    # lr 随 epoch 呈 cos 函数变化， T_max 是周期的一半，eta_min 是 lr 可变化的最小值
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=options['max_epoch'], eta_min=options['lr'] * 1e-3)

    elif options['scheduler'] == 'cosine_warm_restarts':
        # 和 CosineAnnealingLR 相比，CosineAnnealingWarmRestarts 可以调节 lr 的变化周期，使其成倍数 T_mult 的增长；
        # T_0 表示 lr 第一次周期性变化的 epoch 数
        try:
            num_restarts = options['num_restarts']
        except options['num_restarts'].DoesNotExist:
            print('Warning: Num restarts not specified...using 2')
            num_restarts = 2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(options['max_epoch'] / (num_restarts + 1)), eta_min=options['lr'] * 1e-3)

    elif options['scheduler'] == 'cosine_warm_restarts_warmup':
        # 和 CosineAnnealingWarmRestarts 相比，CosineAnnealingWarmupRestartsNew 带有 WarmUp 的部分
        try:
            num_restarts = options['num_restarts']
        except options['num_restarts'].DoesNotExist:
            print('Warning: Num restarts not specified...using 2')
            num_restarts = 2
        scheduler = CosineAnnealingWarmupRestartsNew(
            warmup_epochs=10, optimizer=optimizer, T_0=int(options['max_epoch'] / (num_restarts + 1)),
            eta_min=options['lr'] * 1e-3)

    elif options['scheduler'] == 'warm_restarts_plateau':
        scheduler = WarmRestartPlateau(t_restart=int(options['max_epoch'] / 5),
                                       optimizer=optimizer, threshold_mode='abs',
                                       threshold=0.5, mode='max', patience=int(options['max_epoch'] / 10))

    elif options['scheduler'] == 'multi_step':
        # 阶梯形的 lr 递减，递减比例即为 gamma 参数
        print('Warning: No step list for Multi-Step Scheduler, using constant step of 1/3 epochs')
        steps = [int(options['max_epoch'] * 0.3), int(options['max_epoch'] * 0.6), int(options['max_epoch'] * 0.9)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)

    else:
        raise NotImplementedError

    return scheduler


class WarmRestartPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """    Reduce learning rate on plateau and reset every t_restart epochs     """
    def __init__(self, t_restart, *args, ** kwargs):
        super().__init__(*args, **kwargs)
        self.T_restart = t_restart
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        if self.last_epoch > 0 and self.last_epoch % self.T_restart == 0:

            for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = lr

            self._reset()


class CosineAnnealingWarmupRestartsNew(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, warmup_epochs, *args, **kwargs):

        super(CosineAnnealingWarmupRestartsNew, self).__init__(*args, **kwargs)

        # Init optimizer with low learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min

        self.warmup_epochs = warmup_epochs

        # Get target LR after warmup is complete
        target_lr = (self.eta_min + (self.base_lrs[0] - self.eta_min)
                     * (1 + math.cos(math.pi * warmup_epochs / self.T_i)) / 2)

        # Linearly interpolate between minimum lr and target_lr
        linear_step = (target_lr - self.eta_min) / self.warmup_epochs
        self.warmup_lrs = [self.eta_min + linear_step * (n + 1) for n in range(warmup_epochs)]

    def step(self, epoch=None):             # Called on super class init
        if epoch is None:
            super(CosineAnnealingWarmupRestartsNew, self).step(epoch=epoch)
        else:
            if epoch < self.warmup_epochs:
                lr = self.warmup_lrs[epoch]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Fulfill misc super() funcs
                self.last_epoch = math.floor(epoch)
                self.T_cur = epoch
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            else:
                super(CosineAnnealingWarmupRestartsNew, self).step(epoch=epoch)
