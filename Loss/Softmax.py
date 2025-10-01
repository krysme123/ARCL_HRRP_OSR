import torch.nn as nn
import torch.nn.functional as f
from Loss.LabelSmoothing import smooth_cross_entropy_loss


class Softmax(nn.Module):
    def __init__(self, **options):
        super(Softmax, self).__init__()
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):
        logits = f.softmax(y, dim=1)
        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            loss = f.cross_entropy(y, labels)
        else:
            loss = smooth_cross_entropy_loss(y, labels=labels, smoothing=self.label_smoothing, dim=-1)

        return logits, loss

