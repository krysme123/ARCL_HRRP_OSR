import torch
import torch.nn as nn
import torch.nn.functional as f
from Loss.Dist import Dist
from Loss.LabelSmoothing import smooth_cross_entropy_loss


class RPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(RPLoss, self).__init__()
        self.weight_pl = float(options['lambda'])
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feature_dim'])
        self.radius = 1
        self.radius = nn.Parameter(torch.Tensor(self.radius))
        self.radius.data.fill_(0)
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):
        dist = self.Dist(x)
        logits = f.softmax(dist, dim=1)
        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            loss_main = f.cross_entropy(dist, labels)
        else:
            loss_main = smooth_cross_entropy_loss(dist, labels=labels, smoothing=self.label_smoothing, dim=-1)

        center_batch = self.Dist.centers[labels, :]
        _dis = (x - center_batch).pow(2).mean(1)
        # loss_r = f.mse_loss(_dis, self.radius)
        # 修改后（正确：形状对齐）
        target_radius = self.radius.expand_as(_dis)  # [1] -> [32] (复制标量到相同形状)
        loss_r = f.mse_loss(_dis, target_radius)
        loss = loss_main + self.weight_pl * loss_r

        return logits, loss

    def normalized_logits(self, x):
        dist = self.Dist(x)
        logits = f.softmax(dist, dim=1)
        return logits
