import torch.nn as nn
import torch.nn.functional as f
from Loss.Dist import Dist
from Loss.LabelSmoothing import smooth_cross_entropy_loss


class CenterLoss(nn.Module):
    def __init__(self, **options):
        super(CenterLoss, self).__init__()
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feature_dim'])
        self.weight_pl = options['weight_pl']
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):
        logits = f.softmax(y, dim=1)
        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            softmax_loss = f.cross_entropy(y, labels)
        else:
            softmax_loss = smooth_cross_entropy_loss(y, labels=labels, smoothing=self.label_smoothing, dim=-1)

        center_batch = self.Dist.centers[labels, :]
        center_loss = f.mse_loss(x, center_batch) / 2

        loss = softmax_loss + self.weight_pl * center_loss

        return logits, loss
