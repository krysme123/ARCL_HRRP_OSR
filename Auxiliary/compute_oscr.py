import numpy as np


def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    # correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    # ccr = [0 for x in range(n+2)]
    # fpr = [0 for x in range(n+2)]
    ccr = [0] * (n+2)
    fpr = [0] * (n+2)

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n-1):
        cc = s_k_target[ k +1:].sum()
        fp = s_u_target[k:].sum()
        # True	Positive Rate
        ccr[k] = float(cc) / float(len(x1))
        # False Positive Rate
        fpr[k] = float(fp) / float(len(x2))

    ccr[n] = 0.0
    fpr[n] = 0.0
    ccr[n+1] = 1.0
    fpr[n+1] = 1.0

    # Positions of roc curve (fpr, TPR)
    roc = sorted(zip(fpr, ccr), reverse=True)

    oscr = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range( n +1):
        h =   roc[j][0] - roc[j+1][0]
        w =  (roc[j][1] + roc[j+1][1]) / 2.0
        oscr = oscr + h * w

    return oscr
