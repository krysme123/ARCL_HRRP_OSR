"""
    这个函数是用于 OSR 测试的 超球边界法！
"""
import numpy as np
from Auxiliary.create_logfile import create_boundary_analysis_logfile
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.metrics.pairwise import pairwise_distances


def boundary_hypersphere(train_features, train_labels, test_features, test_labels, out_features, out_labels,
                         **options):
    print("现在开始使用超球边界测试模型的开集性能！\n")
    options['boundary_type'] = 'Boundary_hypersphere'

    # ############################ 计算 auc
    c_proto = np.array([train_features[np.argwhere(train_labels == p).ravel()].mean(0)
                        for p in range(options['num_classes'])])
    all_test_features = np.concatenate((test_features, out_features), axis=0)
    all_test_distance = pairwise_distances(all_test_features, c_proto, metric='euclidean', n_jobs=-1)
    all_test_logits = all_test_distance.min(axis=1)  # 提取每一行的最小值
    all_labels = np.concatenate((np.ones(test_labels.shape[0]), np.zeros(out_labels.shape[0])), axis=0)

    options['auc'] = roc_auc_score(all_labels, -all_test_logits)
    fpr, tpr, _ = roc_curve(all_labels, -all_test_logits)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr', tpr)

    # ############################ 计算 macro_f1
    train_distance = {}
    y_pred = np.zeros((len(options['thresholds']), all_labels.shape[0]))
    y_true = np.concatenate((test_labels, options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)
    options['macro_f1'] = np.zeros(len(options['thresholds']))
    nearest_location = np.argmin(all_test_distance, axis=1)
    distance_range = np.zeros((len(options['thresholds']), options['num_classes']))

    for i in range(len(options['thresholds'])):
        # 先用训练数据做标定
        for j in range(options['num_classes']):
            train_distance[str(j)] = pairwise_distances(
                train_features[np.argwhere(train_labels == j).ravel()], c_proto[j, :].reshape(1, -1),
                metric="euclidean", n_jobs=-1).ravel()
            distance_range[i, j] = np.percentile(train_distance[str(j)], options['thresholds'][i] * 100)

        # 然后依次测试所有待测数据
        for k in range(len(all_labels)):
            if all_test_distance[k, nearest_location[k]] > distance_range[i, nearest_location[k]]:
                y_pred[i, k] = options['num_classes']
            else:
                y_pred[i, k] = nearest_location[k]
        options['macro_f1'][i] = f1_score(y_true, y_pred[i, :], average='macro')

    create_boundary_analysis_logfile(**options)
