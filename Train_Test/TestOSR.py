import numpy as np
import torch
from Train_Test.utils import AverageMeter
from sklearn.metrics import roc_auc_score
from Auxiliary.compute_oscr import compute_oscr


def test_osr(net, criterion, test_loader, out_loader, **options):
    correct, total = 0, 0
    loss_all = 0
    losses = AverageMeter()
    torch.cuda.empty_cache()
    _prediction_k, _prediction_u, _labels_k, _labels_u = [], [], [], []

    with torch.no_grad():
        net.eval()

        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(options['device']), labels.to(options['device'])

            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logic, loss_test = criterion(x, y, labels.long())

                predictions = logic.data.max(1)[1]
                total += labels.size(0)
                correct += predictions.eq(labels.data.view_as(predictions)).sum()

                _prediction_k.append(logic.data.cpu().numpy())
                _labels_k.append(labels.data.cpu().numpy())
            losses.update(loss_test.item(), labels.size(0))
        loss_all += losses.avg

        torch.cuda.empty_cache()
        for batch_idx, (data, labels) in enumerate(out_loader):
            data, labels = data.to(options['device']), labels.to(options['device'])

            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logic, _ = criterion(x, y)
                _prediction_u.append(logic.data.cpu().numpy())

    # Accuracy
    acc = float(correct) / float(total)

    # AUC
    _prediction_k = np.concatenate(_prediction_k, axis=0)
    _prediction_u = np.concatenate(_prediction_u, axis=0)
    _labels_k = np.concatenate(_labels_k, axis=0)

    all_label = np.concatenate((np.ones(_prediction_k.shape[0]), np.zeros(_prediction_u.shape[0])), axis=0)
    all_prediction = np.concatenate((np.max(_prediction_k, axis=1), np.max(_prediction_u, axis=1)), axis=0)
    auc = roc_auc_score(all_label, all_prediction)

    # OSCR
    oscr = compute_oscr(_prediction_k, _prediction_u, _labels_k)

    print('ACC: {:.5f}, AUC: {:.5f}, OSCR: {:.5f}\n'.format(acc, auc, oscr))

    return acc, loss_all, auc, oscr
