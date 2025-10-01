"""
    这个函数是用于 OSR 测试的 原型边界法！
    注意：
        对于含有 prototype 的方法，network 输出的原始 logits 就是基于特征到 prototype 的距离的；
        如果训练方法本身不含有 prototype，那么这个函数调用的就是 基于全连接分类层输出的得分，即 原始性能法；
"""
import numpy as np
from Auxiliary.create_logfile import create_boundary_analysis_logfile
from sklearn.metrics import roc_curve, roc_auc_score, f1_score


def boundary_prototype(train_logits, train_labels, test_logits, test_labels, test_pred_labels,
                       out_logits, out_labels, out_pred_labels, **options):
    print("现在开始使用原型边界（原始性能）测试模型的开集性能！\n")
    options['boundary_type'] = 'Boundary_prototype'

    # ############################ 计算 auc
    all_test_logits = np.concatenate((test_logits, out_logits), axis=0)
    all_labels = np.concatenate((np.ones(test_labels.shape[0]), np.zeros(out_labels.shape[0])), axis=0)

    options['auc'] = roc_auc_score(all_labels, all_test_logits)
    fpr, tpr, _ = roc_curve(all_labels, all_test_logits)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr', tpr)

    # ############################ 计算 macro_f1
    y_pred = np.zeros((len(options['thresholds']), all_labels.shape[0]))
    y_true = np.concatenate((test_labels, options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)
    options['macro_f1'] = np.zeros(len(options['thresholds']))
    logit_range = np.zeros((len(options['thresholds']), options['num_classes']))
    all_pred_labels = np.concatenate((test_pred_labels, out_pred_labels), axis=0)

    for i in range(len(options['thresholds'])):
        # 先用训练数据做标定
        for j in range(options['num_classes']):
            logit_range[i, j] = np.percentile(train_logits[np.argwhere(train_labels == j).astype(int).ravel()],
                                              (1 - options['thresholds'][i]) * 100)
        # 然后依次测试所有待测数据
        for k in range(len(all_labels)):
            if all_test_logits[k] >= logit_range[i, all_pred_labels[k]]:
                y_pred[i, k] = all_pred_labels[k]
            else:
                y_pred[i, k] = options['num_classes']
        options['macro_f1'][i] = f1_score(y_true, y_pred[i, :], average='macro')

    create_boundary_analysis_logfile(**options)
