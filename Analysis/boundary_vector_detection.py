"""
    这个函数是用于 OSR 测试的 向量检测边界法！
    现在已经证实，它和 hypersphere 的结果是一致的！！！
"""
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from Auxiliary.create_logfile import create_boundary_analysis_logfile


def boundary_vector_detection(train_features, train_labels, test_features, test_labels, out_features, out_labels,
                              **options):
    options['boundary_type'] = 'Boundary_vector_detection'

    c_proto = np.array([train_features[np.argwhere(train_labels == p).ravel()].mean(0)
                        for p in range(options['num_classes'])])
    # 计算每个测试样本高维特征到训练样本集特征中心的欧式距离，确定其中最小值所代表的类别
    all_test_features = np.concatenate((test_features, out_features), axis=0)
    test_distance_matrix = pairwise_distances(all_test_features, c_proto, metric='euclidean', n_jobs=-1)
    predicted_class = np.argmin(test_distance_matrix, axis=1)

    # 对每个类别，计算训练样本特征到其余样本特征的向量的和，求欧式长度
    all_labels = np.concatenate((np.ones(test_labels.shape[0]), np.zeros(out_labels.shape[0])), axis=0)
    y_true = np.concatenate((test_labels, options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)
    (train_vector_abs, test_vector_abs, all_test_logits_rearrange, all_labels_rearrange, y_true_rearrange,
     predicted_class_rearrange) = {}, {}, None, None, None, None

    for p in range(options['num_classes']):
        train_features_p = train_features[np.argwhere(train_labels == p)]   # 没有.ravel()，出来维度是 N✖1×F
        clusters_y = np.tile(train_features_p, (1, train_features_p.shape[0], 1))
        clusters_x = np.tile(train_features_p.transpose(1, 0, 2), (train_features_p.shape[0], 1, 1))
        matrix = clusters_x - clusters_y
        vector_sum = np.sum(matrix, axis=1)
        train_vector_abs[str(p)] = np.sqrt(np.sum(vector_sum * vector_sum, axis=1))

        # 计算测试样本高维特征到最近类别的所有训练样本特征的向量和的欧式长度
        all_test_features_p = all_test_features[np.argwhere(predicted_class == p)]
        test_clusters_y = np.tile(all_test_features_p, (1, train_features_p.shape[0], 1))
        train_clusters_x = np.tile(train_features_p.transpose(1, 0, 2), (all_test_features_p.shape[0], 1, 1))
        test_matrix = train_clusters_x - test_clusters_y
        test_vector_sum = np.sum(test_matrix, axis=1)
        test_vector_abs[str(p)] = np.sqrt(np.sum(test_vector_sum * test_vector_sum, axis=1))

        if p == 0:
            all_test_logits_rearrange = test_vector_abs[str(p)]
            all_labels_rearrange = all_labels[np.argwhere(predicted_class == p).ravel()]
            y_true_rearrange = y_true[np.argwhere(predicted_class == p).ravel()]
            predicted_class_rearrange = predicted_class[np.argwhere(predicted_class == p).ravel()]
        else:
            all_test_logits_rearrange = np.append(all_test_logits_rearrange, test_vector_abs[str(p)])
            all_labels_rearrange = np.append(all_labels_rearrange,
                                             all_labels[np.argwhere(predicted_class == p).ravel()])
            y_true_rearrange = np.append(y_true_rearrange, y_true[np.argwhere(predicted_class == p).ravel()])
            predicted_class_rearrange = np.append(predicted_class_rearrange,
                                                  predicted_class[np.argwhere(predicted_class == p).ravel()])

    # 先计算 auc
    options['auc'] = roc_auc_score(all_labels_rearrange, -all_test_logits_rearrange)
    fpr, tpr, _ = roc_curve(all_labels_rearrange, -all_test_logits_rearrange)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr', tpr)

    # 再计算 f1-score
    y_pred = np.zeros((len(options['thresholds']), all_labels.shape[0]))
    options['macro_f1'] = np.zeros(len(options['thresholds']))

    for p in range(len(options['thresholds'])):
        for q in range(len(all_labels)):
            if (all_test_logits_rearrange[q] >
                    np.percentile(train_vector_abs[str(predicted_class_rearrange[q])], options['thresholds'][p]*100)):
                y_pred[p, q] = options['num_classes']
            else:
                y_pred[p, q] = predicted_class_rearrange[q]
        options['macro_f1'][p] = f1_score(y_true_rearrange, y_pred[p, :], average='macro')

    create_boundary_analysis_logfile(**options)
