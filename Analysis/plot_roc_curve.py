
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from Network.VGG32 import VGG32ABN

from Train_Test.utils import AverageMeter
from sklearn.metrics import roc_auc_score
from Auxiliary.compute_oscr import compute_oscr

def plot_roc_curve(net, criterion, test_loader, out_loader, **options):
    correct, total = 0, 0
    loss_all = 0
    losses = AverageMeter()
    torch.cuda.empty_cache()
    _prediction_k, _prediction_u, _labels_k, _labels_u = [], [], [], []

    with torch.no_grad():
        net.eval()

        for data, labels in tqdm(test_loader, desc='Known Scoring'):
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
        for data, labels in tqdm(out_loader, desc='UnKnown Scoring'):
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

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(all_label, all_prediction)
    auc = roc_auc_score(all_label, all_prediction)

    # OSCR
    oscr = compute_oscr(_prediction_k, _prediction_u, _labels_k)

    print('ACC: {:.5f}, AUC: {:.5f}, OSCR: {:.5f}\n'.format(acc, auc, oscr))

    return fpr, tpr, auc


def plot_multiple_roc_curves(paths, models, test_loader, out_loader, loss_names, **options):
    fprs = []
    tprs = []
    aurocs = []
    labels = []

    for path, model_name, loss_name in zip(paths, models, loss_names):
        model = VGG32ABN(options['num_classes']).to(options['device'])

        model.load_state_dict(torch.load(os.path.join(path, 'network', f'{model_name}_100(100).pth'), map_location=options['device']), strict=False)

        # 加载相应的损失函数
        loss_module = importlib.import_module('Loss.' + loss_name)
        criterion = getattr(loss_module, loss_name)(**options).to(options['device'])
        print(f"Loading criterion from {os.path.join(path, 'network', f'{model_name}_100(100)_criterion.pth')}")
        criterion.load_state_dict(torch.load(os.path.join(path, 'network', f'{model_name}_100(100)_criterion.pth'),map_location=options['device']))


        fpr, tpr, auroc = plot_roc_curve(model, criterion, test_loader, out_loader, **options)
        fprs.append(fpr)
        tprs.append(tpr)
        aurocs.append(auroc)
        labels.append(f'{loss_name} (AUROC={auroc:.4f})')

    # 绘制所有ROC曲线
    plt.figure(figsize=(10, 6))
    for fpr, tpr, label in zip(fprs, tprs, labels):
        if label == "ARCL (AUROC=0.7949)":
            plt.plot(fpr, tpr, label=label)
        else:
            plt.plot(fpr, tpr, label=label, linestyle='--')  # 使用虚线

    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)

    # 保存图像到指定路径
    output_dir = os.path.join('./Results_HRRP2D', 'image')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'{options["known"]}_multiple_roc_curves.png'))

    plt.show()