import torch
import numpy as np


def get_features_logits(net, criterion, data_loader, **options):
    features, labels, logits, pred_labels = [], [], [], []
    net.eval()

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.to(options['device'])
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logic, _ = criterion(x, y)
            features.append(x.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())
            logits.append(logic.data.max(1)[0].cpu().detach().numpy())
            pred_labels.append(logic.data.max(1)[1].cpu().detach().numpy())
    features = np.concatenate(features, 0)
    labels = np.concatenate(labels, 0)
    logits = np.concatenate(logits, 0)
    pred_labels = np.concatenate(pred_labels, 0)

    return features, labels, logits, pred_labels
