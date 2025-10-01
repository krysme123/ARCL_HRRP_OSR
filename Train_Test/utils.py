import os
import sys
import errno
import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
# from nested_dict import nested_dict


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value.
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """
    Write console output to external text file.
    
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def load_networks(networks, result_dir, name='', criterion=None, **options):
    filename = '{}/checkpoints/{}/{}.pth'.format(result_dir, options['model'], name)
    networks.load_state_dict(torch.load(filename), False)
    if criterion:
        filename = '{}/checkpoints/{}/{}_criterion.pth'.format(result_dir, options['model'], name)
        criterion.load_state_dict(torch.load(filename))
    return networks, criterion


# WRNs:
def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


# def flatten(params):
#     return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True
