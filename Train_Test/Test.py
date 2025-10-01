import torch
from Train_Test.utils import AverageMeter
import numpy as np


def test(net, criterion, test_loader, **options):
    correct, total = 0, 0
    loss_all = 0
    losses = AverageMeter()
    torch.cuda.empty_cache()

    with torch.no_grad():
        net.eval()

        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(options['device']), labels.to(options['device'])

            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logic, loss_test = criterion(x, y, labels)

                predictions = logic.data.max(1)[1]           # logic 应该就是越大越好了，所以我之前的代码有问题
                total += labels.size(0)
                correct += predictions.eq(labels.data.view_as(predictions)).sum()

            losses.update(loss_test.item(), labels.size(0))
        loss_all += losses.avg

    # Accuracy
    acc = float(correct) / float(total)
    print('Acc: {:.5f}'.format(acc))

    return acc, loss_all
