"""
    这个函数是用于 OSR 测试的 knn-lever-检测边界法！
"""
import numpy as np
from Auxiliary.create_logfile import create_boundary_analysis_logfile
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.metrics.pairwise import pairwise_distances


def boundary_knn_lever_detection(train_features, train_labels, test_features, test_labels, out_features, out_labels,
                              **options):
    print("现在开始使用 knn-lever 检测边界测试模型的开集性能！\n")
    options['boundary_type'] = 'Boundary_knn_lever'

    c_proto = np.array([train_features[np.argwhere(train_labels == p).ravel()].mean(0)
                        for p in range(options['num_classes'])])
    all_test_features = np.concatenate((test_features, out_features), axis=0)

    # ######### 开始检测已知特征的边界
    distance_matrix = pairwise_distances(train_features, train_features, metric='euclidean', n_jobs=-1)
    k_index = np.argsort(distance_matrix)[:, 1:1 + options['k']]
    k_objects = train_features[k_index]
    repeat_features = np.repeat(train_features.reshape([train_features.shape[0], 1, -1]), options['k'], axis=1)
    term =\
        np.tanh(np.linalg.norm(np.sum(k_objects - repeat_features, axis=1), ord=1, axis=1))\
        * np.tanh(np.linalg.norm(np.sum(k_objects - repeat_features, axis=1), ord=2, axis=1))
    term_index = np.argsort(-term)

    all_test_distance = pairwise_distances(all_test_features, c_proto, metric='euclidean', n_jobs=-1)
    location = np.argmin(all_test_distance, axis=1)

    test_distance_matrix = pairwise_distances(all_test_features, train_features, metric='euclidean', n_jobs=-1)
    test_k_index = np.argsort(test_distance_matrix)[:, 1:1 + options['k']]
    test_k_objects = train_features[test_k_index]
    repeat_features = np.repeat(all_test_features.reshape([all_test_features.shape[0], 1, -1]), options['k'], axis=1)
    test_term =\
        np.tanh(np.linalg.norm(np.sum(test_k_objects - repeat_features, axis=1), ord=1, axis=1))\
        * np.tanh(np.linalg.norm(np.sum(test_k_objects - repeat_features, axis=1), ord=2, axis=1))

    all_labels = np.concatenate((np.ones(test_labels.shape[0]), np.zeros(out_labels.shape[0])), axis=0)

    options['auc'] = roc_auc_score(all_labels, -test_term)
    fpr, tpr, _ = roc_curve(all_labels, -test_term)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr', tpr)

    y_pred = np.zeros((len(options['thresholds']), all_labels.shape[0]))
    y_true = np.concatenate((test_labels, options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)
    options['macro_f1'] = np.zeros(len(options['thresholds']))

    for p in range(len(options['thresholds'])):
        for q in range(len(all_labels)):
            if test_term[q] > term[term_index[round((1-float(options['thresholds'][p]))*term.shape[0])]]:
                y_pred[p, q] = options['num_classes']
            else:
                y_pred[p, q] = location[q]
        options['macro_f1'][p] = f1_score(y_true, y_pred[p, :], average='macro')

    create_boundary_analysis_logfile(**options)
