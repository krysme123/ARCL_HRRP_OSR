import torch.nn as nn
import torch.nn.functional as f
from Loss.Dist import Dist
from Loss.LabelSmoothing import smooth_cross_entropy_loss


class GCPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(GCPLoss, self).__init__()
        self.weight_pl = options['weight_pl']
        # self.temp = options['temp']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feature_dim'])
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):
        dist = self.Dist(x)
        logits = f.softmax(-dist, dim=1)
        if labels is None:
            return logits, 0
        # loss = f.cross_entropy(-dist / self.temp, labels)

        if not self.label_smoothing:
            loss = f.cross_entropy(-dist, labels)
        else:
            loss = smooth_cross_entropy_loss(-dist, labels=labels, smoothing=self.label_smoothing, dim=-1)

        center_batch = self.Dist.centers[labels, :]
        loss_r = f.mse_loss(x, center_batch) / 2
        loss = loss + self.weight_pl * loss_r
        return logits, loss

    def normalized_logits(self, x):
        dist = self.Dist(x)
        logits = f.softmax(-dist, dim=1)
        return logits
