from torch import nn


class _MultiBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MultiBatchNorm, self).__init__()
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label


class MultiBatchNorm(_MultiBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


def weights_init(m):
    classname = m.__class__.__name__    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_ABN(m):
    classname = m.__class__.__name__    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('MultiBatchNorm') != -1:
        m.bns[0].weight.data.normal_(1.0, 0.02)
        m.bns[0].bias.data.fill_(0)
        m.bns[1].weight.data.normal_(1.0, 0.02)
        m.bns[1].bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _MultiBatchNorm1D(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MultiBatchNorm1D, self).__init__()
        # self.bns =
        # nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats)
        # for _ in range(num_classes)])
        # self.bns = nn.ModuleList(
        #     [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input_):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label


class MultiBatchNorm1D(_MultiBatchNorm1D):
    def _check_input_dim(self, input_):
        # if input_.dim() != 4:             # 2D data should use this "if"
        #     raise ValueError('expected 4D input (got {}D input)'.format(input_.dim()))
        if input_.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input_.dim()))
