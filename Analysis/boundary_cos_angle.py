"""
    这个函数是用于 OSR 测试的 向量夹角检测边界法！
"""
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from Auxiliary.create_logfile import create_boundary_analysis_logfile
from Analysis.plot_confusion_matrix import plot_confusion_matrix
from Dataset.HRRP_datasets import HRRPOSRData310
from torch.utils.data import DataLoader


def boundary_cos_angle(train_features, train_labels, test_features, test_labels, out_features, out_labels, **options):
    options['boundary_type'] = 'Boundary_cos_angle'

    c_proto = np.array([train_features[np.argwhere(train_labels == p).ravel()].mean(0)
                        for p in range(options['num_classes'])])

    all_test_features = np.concatenate((test_features, out_features), axis=0)
    test_distance_matrix = pairwise_distances(all_test_features, c_proto, metric='euclidean', n_jobs=-1)
    predicted_class = np.argmin(test_distance_matrix, axis=1)

    all_labels = np.concatenate((np.ones(test_labels.shape[0]), np.zeros(out_labels.shape[0])), axis=0)
    y_true = np.concatenate((test_labels, options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)
    y_true_cm = np.concatenate((test_labels, out_labels+options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)

    all_test_logits_rearrange, all_labels_rearrange, y_true_rearrange, predicted_class_rearrange, y_true_cm_rearrange =\
        None, None, None, None, None

    train_p_value, test_p_value = {}, {}              # cos 夹角数值的和，越大越是边界点

    for p in range(options['num_classes']):
        train_features_p = train_features[np.argwhere(train_labels == p)]   # 没有.ravel()，出来维度是 N✖1×F
        clusters_y = np.tile(train_features_p, (1, train_features_p.shape[0], 1))
        clusters_x = np.tile(train_features_p.transpose(1, 0, 2), (train_features_p.shape[0], 1, 1))
        train_features_matrix = clusters_x - clusters_y

        c_proto_matrix = np.tile(c_proto[p, :][np.newaxis, :], (train_features_p.shape[0], 1))
        train_features_proto_matrix = c_proto_matrix - np.squeeze(train_features_p)

        cos_angle = np.zeros(train_features_p.shape[0])
        for i in range(train_features_p.shape[0]):
            cos_angle[i] = np.mean(
                np.dot(train_features_proto_matrix[i, :] /
                       (np.linalg.norm(train_features_proto_matrix[i, :]) + np.finfo(np.float64).eps),
                       np.transpose(train_features_matrix[i, :, :] /
                                    (np.linalg.norm(train_features_matrix[i, :, :], axis=1, keepdims=True)
                                     + np.finfo(np.float64).eps)))
            )
        train_p_value[str(p)] = cos_angle

        all_test_features_p = all_test_features[np.argwhere(predicted_class == p)]
        test_clusters_y = np.tile(all_test_features_p, (1, train_features_p.shape[0], 1))
        train_clusters_x = np.tile(train_features_p.transpose(1, 0, 2), (all_test_features_p.shape[0], 1, 1))
        test_features_matrix = train_clusters_x - test_clusters_y

        c_proto_matrix_test = np.tile(c_proto[p, :][np.newaxis, :], (all_test_features_p.shape[0], 1))
        test_features_proto_matrix = c_proto_matrix_test - np.squeeze(all_test_features_p)

        cos_angle_test = np.zeros(all_test_features_p.shape[0])
        for i in range(all_test_features_p.shape[0]):
            cos_angle_test[i] = np.mean(
                np.dot(test_features_proto_matrix[i, :] /
                       (np.linalg.norm(test_features_proto_matrix[i, :]) + np.finfo(np.float64).eps),
                       np.transpose(test_features_matrix[i, :, :] /
                                    (np.linalg.norm(test_features_matrix[i, :, :], axis=1, keepdims=True))
                                    + np.finfo(np.float64).eps))
            )
        test_p_value[str(p)] = cos_angle_test

        if p == 0:
            all_test_logits_rearrange = test_p_value[str(p)]
            all_labels_rearrange = all_labels[np.argwhere(predicted_class == p).ravel()]
            y_true_rearrange = y_true[np.argwhere(predicted_class == p).ravel()]
            predicted_class_rearrange = predicted_class[np.argwhere(predicted_class == p).ravel()]
            y_true_cm_rearrange = y_true_cm[np.argwhere(predicted_class == p).ravel()]
        else:
            all_test_logits_rearrange = np.append(all_test_logits_rearrange, test_p_value[str(p)])
            all_labels_rearrange = np.append(all_labels_rearrange,
                                             all_labels[np.argwhere(predicted_class == p).ravel()])
            y_true_rearrange = np.append(y_true_rearrange, y_true[np.argwhere(predicted_class == p).ravel()])
            predicted_class_rearrange = np.append(predicted_class_rearrange,
                                                  predicted_class[np.argwhere(predicted_class == p).ravel()])
            y_true_cm_rearrange = np.append(y_true_cm_rearrange, y_true_cm[np.argwhere(predicted_class == p).ravel()])

    # 先计算 auc
    options['auc'] = roc_auc_score(all_labels_rearrange, -all_test_logits_rearrange)
    fpr, tpr, _ = roc_curve(all_labels_rearrange, -all_test_logits_rearrange)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr', tpr)

    # 再计算 f1-score
    y_pred = np.zeros((len(options['thresholds']), all_labels.shape[0]))
    options['macro_f1'] = np.zeros(len(options['thresholds']))

    cm_y_classes = options['known_classes']+options['unknown_classes']
    cm_x_classes = options['known_classes']+['Unknown']

    for p in range(len(options['thresholds'])):
        for q in range(len(all_labels)):
            if (all_test_logits_rearrange[q] >
                    np.percentile(train_p_value[str(predicted_class_rearrange[q])], options['thresholds'][p]*100)):
                y_pred[p, q] = options['num_classes']
            else:
                y_pred[p, q] = predicted_class_rearrange[q]
        options['macro_f1'][p] = f1_score(y_true_rearrange, y_pred[p, :], average='macro')

        plot_confusion_matrix(y_true=y_true_cm_rearrange, y_pred=y_pred[p, :], y_classes=cm_y_classes,
                              x_classes=cm_x_classes,
                              threshold=options['thresholds'][p], rotation=True, **options)

    create_boundary_analysis_logfile(**options)
