import torch
import torch.nn as nn
import torch.nn.functional as f
from Loss.Dist import Dist
from Loss.LabelSmoothing import smooth_cross_entropy_loss


class SLCPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(SLCPLoss, self).__init__()
        self.use_gpu = options['use_gpu']
        self.weight_pl = float(options['lambda'])
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feature_dim'])
        self.points = self.Dist.centers
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):
        dist_l2_p = self.Dist(x, center=self.points)
        logits = f.softmax(-dist_l2_p, dim=1)
        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            loss_main = f.cross_entropy(-dist_l2_p, labels)
        else:
            loss_main = smooth_cross_entropy_loss(-dist_l2_p, labels=labels, smoothing=self.label_smoothing, dim=-1)

        center_batch = self.points[labels, :]
        loss_r = f.mse_loss(x, center_batch) / 2

        o_center = self.points.mean(0)
        l_ = (self.points - o_center).pow(2).mean(1)
        # loss_outer = torch.exp(-l_.mean(0))
        loss_outer_std = torch.std(l_)

        loss = loss_main + self.weight_pl * loss_r + loss_outer_std
        return logits, loss

    def normalized_logits(self, x):
        dist_l2_p = self.Dist(x, center=self.points)
        logits = f.softmax(-dist_l2_p, dim=1)
        return logits
