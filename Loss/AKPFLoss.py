import torch
import torch.nn as nn
import torch.nn.functional as f
from Loss.Dist import Dist
from Loss.LabelSmoothing import smooth_cross_entropy_loss


class AKPFLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(AKPFLoss, self).__init__()
        self.use_gpu = options['use_gpu']
        self.weight_pl = float(options['weight_pl'])
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feature_dim'])
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=0)
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        logits = -1*(dist_l2_p - dist_dot_p)           # [batch_size, number_class]

        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            loss_main = f.cross_entropy(logits, labels)
        else:
            loss_main = smooth_cross_entropy_loss(logits, labels=labels, smoothing=self.label_smoothing, dim=-1)

        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r = self.margin_loss(self.radius, _dis_known, target)
        # loss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)    max(0, d_e(C(x), P^k) - R + 1)

        loss = loss_main + self.weight_pl * loss_r
        return logits, loss

    def normalized_logits(self, x):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        logits = -1*(dist_l2_p - dist_dot_p)           # [batch_size, number_class]
        return f.softmax(logits, dim=1)
