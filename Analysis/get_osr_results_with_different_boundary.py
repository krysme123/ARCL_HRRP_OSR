from Analysis.get_features_logits import get_features_logits
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, auc, f1_score
from Analysis.plot_confusion_matrix import plot_confusion_matrix
import os
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from scipy import stats
from Auxiliary.create_dir_file import name_transport


#  原型边界，超球边界，极值边界，检测边界，新检测边界

#  在调用已有特征提取模型的基础上，结合上述边界，得到相应的osr的实验结果： auc，oscr，f1-score(threshold)


def get_osr_results_with_different_boundary(net, criterion, train_loader, test_loader, out_loader, thresholds,
                                            save_path, **options):
    """         没有原型边界的则代表的是他自身方法的 logtis，如 softmax，Ring 等
    :param net:
    :param criterion:
    :param train_loader:
    :param test_loader:
    :param out_loader:
    :param thresholds:          for examples: [0.1, 0.2, ..., 0.8, 0.9, 0.92, 0.94]
    :param save_path:
    :param options:
    :return:
    """
    logic_range = np.zeros((len(thresholds), options['num_classes']))
    results = {}

    features_k, labels_k, logits_k, pred_labels_k = get_features_logits(net, criterion, train_loader, **options)
    features_k_test,  labels_k_test, logits_k_test, pred_labels_k_test =\
        get_features_logits(net, criterion, test_loader, **options)
    features_u,  labels_u, logits_u, pred_labels_u = get_features_logits(net, criterion, out_loader, **options)

    y_true = np.concatenate((labels_k_test, options['num_classes'] * np.ones(labels_u.shape[0])), axis=0)
    y_true_plot_confusion_matrix = np.concatenate((labels_k_test, options['num_classes'] + labels_u))

    logits = np.concatenate((logits_k_test, logits_u), axis=0)
    pred_labels = np.concatenate((pred_labels_k_test, pred_labels_u), axis=0)
    y_pred = np.zeros((len(thresholds), pred_labels.shape[0]))
    y_one_zero = np.concatenate((np.ones(labels_k_test.shape[0]), np.zeros(labels_u.shape[0])), axis=0)

    features = np.concatenate((features_k_test, features_u), axis=0)

    distance_features, c_proto, all_test_distance, distance_matrix = {}, None, {}, None
    name = name_transport(**options)

    # ################################################### 第一种方法：原始（原型）边界  ##############################
    if options['boundary'] == 'prototype' or options['boundary'] == 'all':
        print("使用第一种边界方法进行评估：原始（原型）边界\n")
        confusion_matrix_dir_prototype = save_path + '/confusion_matrix_image_prototype'
        if not os.path.exists(confusion_matrix_dir_prototype):
            os.makedirs(confusion_matrix_dir_prototype)

        for j in range(len(thresholds)):
            for k in range(options['num_classes']):
                logic_range[j, k] = np.percentile(logits_k[np.argwhere(labels_k == k).astype(int).ravel()],
                                                  (1 - thresholds[j]) * 100)
        for q in range(len(thresholds)):
            for p in range(y_pred.shape[0]):
                if logits[p] >= logic_range[q, pred_labels[p]]:
                    y_pred[q, p] = pred_labels[p]
                else:
                    y_pred[q, p] = options['num_classes']

            results['prototype_macro_average_f1_' + str(thresholds)] = f1_score(y_true, y_pred[q, :], average='macro')
            results['prototype_weighted_f1_' + str(thresholds)] = f1_score(y_true, y_pred[q, :], average='weighted')

            plot_confusion_matrix(y_true_plot_confusion_matrix, y_pred[q, :], options['y_classes'],
                                  options['x_classes'], save_path=confusion_matrix_dir_prototype,
                                  threshold=thresholds[q], rotation=True, **options)
        fpr, tpr, _ = roc_curve(y_one_zero, logits)  # 库内的 logits 越大越好
        results['prototype_auc'] = auc(fpr, tpr)

    # ############################################# 第二种方法：超球边界 ###########################################
    if options['boundary'] == 'hypersphere' or options['boundary'] == 'all':
        print("使用第二种边界方法进行评估：超球边界\n")
        confusion_matrix_dir_hypersphere = save_path + '/confusion_matrix_image_hypersphere'
        if not os.path.exists(confusion_matrix_dir_hypersphere):
            os.makedirs(confusion_matrix_dir_hypersphere)

        c_proto = np.array([features_k[np.argwhere(labels_k == p).ravel()].mean(0)
                            for p in range(options['num_classes'])])
        for p in range(options['num_classes']):
            distance_features[str(p)] = pairwise_distances(
                features_k[np.argwhere(labels_k == p).ravel()], c_proto[p, :].reshape(1, -1),
                metric="euclidean", n_jobs=-1).ravel()

        all_test_distance = pairwise_distances(features, c_proto, metric='euclidean', n_jobs=-1)
        logits_distance = all_test_distance.min(axis=1)             # 提取每一行的最小值
        nearest_location = all_test_distance.argmin(axis=1)         # 提取每一行的最小值的索引

        fpr, tpr, _ = roc_curve(y_one_zero, -1 * logits_distance)  # 库内的 logits 越大越好
        results['hypersphere_auc'] = auc(fpr, tpr)

        distance_range = np.zeros((len(thresholds), options['num_classes']))
        for j in range(options['num_classes']):
            for k in range(len(thresholds)):
                distance_range[k, j] = np.percentile(distance_features[str(j)], thresholds[k] * 100)

        for q in range(len(thresholds)):
            for p in range(y_pred.shape[0]):
                if all_test_distance[p, nearest_location[p]] >= distance_range[q, nearest_location[p]]:
                    y_pred[q, p] = options['num_classes']
                else:
                    y_pred[q, p] = nearest_location[p]
            results['hypersphere_macro_average_f1_' + str(thresholds)] = f1_score(y_true, y_pred[q, :], average='macro')
            results['hypersphere_weighted_f1_' + str(thresholds)] = f1_score(y_true, y_pred[q, :], average='weighted')

            plot_confusion_matrix(y_true_plot_confusion_matrix, y_pred[q, :], options['y_classes'],
                                  options['x_classes'], save_path=confusion_matrix_dir_hypersphere,
                                  threshold=thresholds[q], rotation=True, **options)

    # ############################################### 第三种方法：极值边界 ###############################################
    if options['boundary'] == 'gev' or options['boundary'] == 'all':
        print("使用第三种边界方法进行评估：极值边界\n")
        confusion_matrix_dir_gev = save_path + '/confusion_matrix_image_gev'
        if not os.path.exists(confusion_matrix_dir_gev):
            os.makedirs(confusion_matrix_dir_gev)

        all_logic = np.zeros(y_pred.shape[0])
        if options['boundary'] == 'gev':
            c_proto = np.array([features_k[np.argwhere(labels_k == p).ravel()].mean(0)
                                for p in range(options['num_classes'])])
            all_test_distance = pairwise_distances(features, c_proto, metric='euclidean', n_jobs=-1)

        num_bins = 50
        pdf_x, pdf_y = np.zeros((options['num_classes'], num_bins + 1)), np.zeros((options['num_classes'], num_bins))
        cdf_x, cdf_y = np.zeros((options['num_classes'], num_bins + 1)), np.zeros((options['num_classes'], num_bins))
        x, popt = np.zeros((options['num_classes'], 1000)), np.zeros((options['num_classes'], 2))
        parameters = np.zeros((options['num_classes'], 3))

        options['gev_dir'] = save_path + '/GEV_Fitting'
        if not os.path.exists(options['gev_dir']):
            os.makedirs(options['gev_dir'])

        for p in range(options['num_classes']):
            if options['boundary'] == 'extreme':
                distance_features[str(p)] = pairwise_distances(
                    features_k[np.argwhere(labels_k == p).ravel()], c_proto[p, :].reshape(1, -1),
                    metric="euclidean", n_jobs=-1).ravel()
            plt.figure()
            c, loc, scale = gev.fit(distance_features[str(p)])
            parameters[p, 0], parameters[p, 1], parameters[p, 2] = c, loc, scale
            pdf_y[p, :], pdf_x[p, :], _ = plt.hist(distance_features[str(p)], histtype='barstacked', density=True,
                                                   bins=num_bins, edgecolor='black')
            plt.plot(pdf_x[p, :], gev.pdf(pdf_x[p, :], c, loc, scale),
                     label=r'GEV_PDF: $\xi={:.3f}, \alpha={:.3f}, \beta={:.3f}$'.format(c, loc, scale))
            plt.xlabel(r'$||\Theta(x)-O||_2$')
            plt.ylabel('Probability density')
            plt.legend()
            plt.savefig(options['gev_dir'] + '/' + name + '_GEV_PDF_{}.pdf'.format(p))
            plt.close()
            plt.figure()
            cdf_y[p, :], cdf_x[p, :], _ = plt.hist(distance_features[str(p)], histtype='barstacked', density=True,
                                                   cumulative=True, bins=num_bins, edgecolor='black')
            statistic, pvalue = stats.ks_2samp(cdf_y[p, :], gev.cdf(cdf_x[p, :], c, loc, scale))
            x[p, :] = np.linspace(cdf_x[p, 0], cdf_x[p, -1], 1000)
            plt.plot(x[p, :], gev.cdf(x[p, :], c, loc, scale), label='GEV_CDF: $P={:.3f}$'.format(pvalue))
            plt.xlabel(r'$||\Theta(x)-O||_2$')
            plt.ylabel('Probability')
            plt.legend()
            plt.savefig(options['gev_dir'] + '/' + name + '_GEV_CDF_{}.pdf'.format(p))
            plt.close()

        for q in range(len(thresholds)):
            for p in range(y_pred.shape[0]):
                location = 0
                all_logic[p] = gev.cdf(all_test_distance[p, location], parameters[location, 0], parameters[location, 1],
                                       parameters[location, 2])
                for pp in np.arange(1, options['num_classes']):
                    new_logic = gev.cdf(all_test_distance[p, pp], parameters[pp, 0], parameters[pp, 1],
                                        parameters[pp, 2])
                    if all_logic[p] > new_logic:
                        all_logic[p] = new_logic
                        location = pp

                if all_logic[p] >= thresholds[q]:
                    y_pred[q, p] = options['num_classes']
                else:
                    y_pred[q, p] = location
            results['gev_macro_average_f1_' + str(thresholds)] = f1_score(y_true, y_pred[q, :], average='macro')
            results['gev_weighted_f1_' + str(thresholds)] = f1_score(y_true, y_pred[q, :], average='weighted')

            plot_confusion_matrix(y_true_plot_confusion_matrix, y_pred[q, :], options['y_classes'],
                                  options['x_classes'], save_path=confusion_matrix_dir_gev,
                                  threshold=thresholds[q], rotation=True, **options)

        fpr, tpr, _ = roc_curve(y_one_zero, -1 * all_logic)
        results['gev_auc'] = auc(fpr, tpr) * 100

    # ############################################### 第四种方法：lever检测边界 ##########################################
    if options['boundary'] == 'lever' or options['boundary'] == 'all':
        print("使用第四种边界方法进行评估：lever检测边界\n")
        confusion_matrix_dir_lever = save_path + '/confusion_matrix_image_lever'
        if not os.path.exists(confusion_matrix_dir_lever):
            os.makedirs(confusion_matrix_dir_lever)

        k = options['k_lever']
        if options['boundary'] == 'lever':
            distance_matrix = pairwise_distances(features_k, features_k, metric='euclidean', n_jobs=-1)
            all_test_distance = pairwise_distances(features, c_proto, metric='euclidean', n_jobs=-1)

        k_index = np.argsort(distance_matrix)[:, 1:1 + k]
        k_objects = features_k[k_index]

        repeat_features = np.repeat(features_k.reshape([features_k.shape[0], 1, -1]), k, axis=1)
        term = \
            np.tanh(np.linalg.norm(np.sum(k_objects - repeat_features, axis=1), ord=1, axis=1)) \
            * np.tanh(np.linalg.norm(np.sum(k_objects - repeat_features, axis=1), ord=2, axis=1))
        term_index = np.argsort(-term)
        location = np.argmin(all_test_distance, axis=1)

        test_distance_matrix = pairwise_distances(features, features_k, metric='euclidean', n_jobs=-1)
        test_k_index = np.argsort(test_distance_matrix)[:, 1:1 + k]
        test_k_objects = features_k[test_k_index]
        repeat_features = np.repeat(features.reshape([features.shape[0], 1, -1]), k, axis=1)
        test_term = \
            np.tanh(np.linalg.norm(np.sum(test_k_objects - repeat_features, axis=1), ord=1, axis=1)) \
            * np.tanh(np.linalg.norm(np.sum(test_k_objects - repeat_features, axis=1), ord=2, axis=1))

        for q in range(len(thresholds)):
            for p in range(y_true.shape[0]):
                if test_term[p] > term[term_index[round((1 - float(thresholds[q])) * term.shape[0])]]:
                    y_pred[q, p] = options['num_classes']
                else:
                    y_pred[q, p] = location[p]

            results['lever_macro_average_f1_' + str(thresholds)] = f1_score(y_true, y_pred[q, :], average='macro')
            results['lever_weighted_f1_' + str(thresholds)] = f1_score(y_true, y_pred[q, :], average='weighted')

            plot_confusion_matrix(y_true_plot_confusion_matrix, y_pred[q, :], options['y_classes'],
                                  options['x_classes'], save_path=confusion_matrix_dir_lever,
                                  threshold=thresholds[q], rotation=True, **options)

        fpr, tpr, _ = roc_curve(y_one_zero, -1 * test_term)  # 库内的 logits 越大越好
        results['lever_auc'] = auc(fpr, tpr)

    # ############################################### 第五种方法：向量检测边界 ##########################################
