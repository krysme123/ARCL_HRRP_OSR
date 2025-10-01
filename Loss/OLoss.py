import torch
import torch.nn as nn
import torch.nn.functional as f
from Loss.LabelSmoothing import smooth_cross_entropy_loss
from Loss.Dist import Dist


class OLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(OLoss, self).__init__()
        self.weight_pl = options['weight_pl']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feature_dim'])
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(1)
        self.points = self.Dist.centers
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)
        self.device = options['device']
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):
        dist_l2_p = self.Dist(x, center=self.points)  # formula (6):d_e

        o_center = torch.mean(self.Dist.centers, dim=0)
        dist_dot_p = self.Dist(x-o_center.repeat(x.size(0), 1),
                               center=self.points-o_center.repeat(self.Dist.num_classes, 1),
                               metric='dot')  # formula (6):d_d
        logits = dist_l2_p - dist_dot_p  # [batch_size,num_classes]

        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            loss_main = f.cross_entropy(logits, labels)
        else:
            loss_main = smooth_cross_entropy_loss(logits, labels=labels, smoothing=self.label_smoothing, dim=-1)

        _dis_known = (x - o_center.repeat(x.size(0), 1)).pow(2).mean(1)
        # target = torch.ones(_dis_known.size()).cuda()
        target = torch.ones(_dis_known.size()).to(self.device)

        loss_r1 = self.margin_loss(self.radius, _dis_known, target)
        loss_r2 = self.margin_loss(_dis_known, self.radius, target)
        # loss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)    max(0, d_e(C(x), P^k) - R + 1)

        loss = loss_main + self.weight_pl * (loss_r1 + loss_r2)
        return logits, loss

    def normalized_logits(self, x):
        dist_l2_p = self.Dist(x, center=self.points)  # formula (6):d_e
        o_center = torch.mean(self.Dist.centers, dim=0)
        dist_dot_p = self.Dist(x-o_center.repeat(x.size(0), 1),
                               center=self.points-o_center.repeat(self.Dist.num_classes, 1),
                               metric='dot')  # formula (6):d_d
        logits = dist_l2_p - dist_dot_p  # [batch_size,num_classes]
        return f.softmax(logits, dim=1)
