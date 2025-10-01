"""
    这个函数是用于 OSR 测试的 最小凸包边界法！

    经过测试，得到出错信息如下：
    scipy.spatial._qhull.QhullError: QH6154 Qhull precision error:
        Initial simplex is flat (facet 1 is coplanar with the interior point)

    失败！！！
        原因是输出的特征不满秩，高维空间的稀疏性！

    经过调整后，使用 t-SNE 技术，统一降维之后，再使用凸包检测就可以了！
"""
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from Auxiliary.create_logfile import create_boundary_analysis_logfile
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from scipy.optimize import minimize


def points_in_hull(points: np.ndarray, hull: ConvexHull, tolerance: float = 1e-12):
    return np.all(np.add(points @ hull.equations[:, :-1].T, hull.equations[:, -1]) <= tolerance, axis=1)


def compute(target, convex_hull):
    ll = convex_hull.shape[0]

    def obj(x):
        result = target - sum(np.dot(np.diag(x), convex_hull))
        return np.linalg.norm(result)
    # 不等式约束
    ineq_cons = {"type": "ineq",
                 "fun": lambda x: x}
    # 等式约束
    eq_cons = {"type": "eq",
               "fun": lambda x: sum(x)-1}
    x0 = np.ones(ll)/ll
    res = minimize(obj, x0, method='SLSQP', constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': False})
    return res.fun


def boundary_convexhull(train_features, train_labels, test_features, test_labels, out_features, out_labels,
                        **options):
    options['boundary_type'] = 'Boundary_convexhull'

    c_proto = np.array([train_features[np.argwhere(train_labels == p).ravel()].mean(0)
                        for p in range(options['num_classes'])])

    # 计算每个测试样本高维特征到训练样本集特征中心的欧式距离，确定其中最小值所代表的类别
    all_test_features = np.concatenate((test_features, out_features), axis=0)
    test_distance_matrix = pairwise_distances(all_test_features, c_proto, metric='euclidean', n_jobs=-1)
    predicted_class = np.argmin(test_distance_matrix, axis=1)
    all_labels = np.concatenate((np.ones(test_labels.shape[0]), np.zeros(out_labels.shape[0])), axis=0)
    y_true = np.concatenate((test_labels, options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)

    all_hull = {}
    for i in range(options['num_classes']):
        train_features_i = train_features[np.argwhere(train_labels == i).ravel()]
        all_hull[str(i)] = ConvexHull(train_features_i)

    # 计算 f1-score，注意，凸包边界方法，不存在 thresholds 的问题！！！！！
    y_pred = np.zeros(all_labels.shape[0])
    y_pred_auc = np.zeros(all_labels.shape[0])

    for q in range(len(all_labels)):
        in_hull = points_in_hull(all_test_features[q], all_hull[str(predicted_class[q])])
        if in_hull:
            y_pred[q] = predicted_class[q]
            y_pred_auc[q] = 1
        else:
            y_pred[q] = options['num_classes']
            y_pred_auc[q] = 0
    options['macro_f1'] = f1_score(y_true, y_pred, average='macro')

    # 计算 auc
    options['auc'] = roc_auc_score(all_labels, y_pred_auc)
    fpr, tpr, _ = roc_curve(all_labels, y_pred_auc)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr', tpr)

    create_boundary_analysis_logfile(**options)


def boundary_tsne_convexhull(train_features, train_labels, test_features, test_labels, out_features, out_labels,
                             **options):
    options['boundary_type'] = 'Boundary_tsne_convexhull'

    tsne = TSNE(n_components=3, init='pca', random_state=0)
    all_train_test_features = np.concatenate((train_features, test_features, out_features), axis=0)
    all_tsne_features = tsne.fit_transform(all_train_test_features)
    train_numbers = len(train_labels)

    c_proto = np.array([all_tsne_features[np.argwhere(train_labels == p).ravel()].mean(0)
                        for p in range(options['num_classes'])])
    test_distance_matrix = pairwise_distances(all_tsne_features[train_numbers:], c_proto, metric='euclidean', n_jobs=-1)
    predicted_class = np.argmin(test_distance_matrix, axis=1)

    all_hull, tsne_train_features = {}, {}
    for i in range(options['num_classes']):
        tsne_train_features[str(i)] = all_tsne_features[np.argwhere(train_labels == i).ravel()]
        all_hull[str(i)] = ConvexHull(tsne_train_features[str(i)])

    # 计算 f1-score，注意，凸包边界方法，不存在 thresholds 的问题！！！！！
    all_labels = np.concatenate((np.ones(test_labels.shape[0]), np.zeros(out_labels.shape[0])), axis=0)
    y_pred = np.zeros(all_labels.shape[0])
    y_true = np.concatenate((test_labels, options['num_classes'] * np.ones(out_labels.shape[0])), axis=0)
    y_pred_auc_1 = np.zeros(all_labels.shape[0])
    y_pred_auc_2 = np.zeros(all_labels.shape[0])
    y_pred_auc_3 = np.zeros(all_labels.shape[0])

    for q in range(len(all_labels)):
        in_hull = points_in_hull(all_tsne_features[q+train_numbers, :].reshape(1, -1),
                                 all_hull[str(predicted_class[q])])
        if in_hull:
            y_pred[q] = predicted_class[q]
            y_pred_auc_1[q] = 1
            y_pred_auc_2[q] = 1
            y_pred_auc_3[q] = 1
        else:
            y_pred[q] = options['num_classes']
            y_pred_auc_1[q] = 0
            y_pred_auc_2[q] = np.exp(-np.min(test_distance_matrix[q]))
            y_pred_auc_3[q] = np.exp(-compute(all_tsne_features[q+train_numbers, :],
                                              tsne_train_features[str(predicted_class[q])]
                                              [all_hull[str(predicted_class[q])].vertices]))

    options['macro_f1'] = f1_score(y_true, y_pred, average='macro')

    # 计算 auc
    auc_1 = roc_auc_score(all_labels, y_pred_auc_1)
    auc_2 = roc_auc_score(all_labels, y_pred_auc_2)
    auc_3 = roc_auc_score(all_labels, y_pred_auc_3)
    options['auc'] = np.array([auc_1, auc_2, auc_3])
    if np.argmax(options['auc']) == 0:
        fpr, tpr, _ = roc_curve(all_labels, y_pred_auc_1)
    elif np.argmax(options['auc']) == 1:
        fpr, tpr, _ = roc_curve(all_labels, y_pred_auc_2)
    else:
        fpr, tpr, _ = roc_curve(all_labels, y_pred_auc_3)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_fpr', fpr)
    np.save(options['save_path'] + '/' + options['boundary_type'] + '_tpr', tpr)

    create_boundary_analysis_logfile(**options)
