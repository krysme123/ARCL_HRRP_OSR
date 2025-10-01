"""
    这个是验证聚类形变问题的子程序，把 net、criterion、data 等信息传进来，输出相应的聚类形变度量参数！
"""
from Analysis.get_features_logits import get_features_logits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


def compute_cluster_deformation_parameter(net, criterion, data_loader, **options):
    data_features, data_labels, data_logits, pred_labels = get_features_logits(net, criterion, data_loader, **options)
    o_center = np.mean(data_features, axis=0)
    i_features = {}
    results = np.zeros((options['num_classes'], 4))
    for i in range(options['num_classes']):
        location = np.argwhere(data_labels == i).ravel()
        i_features[i] = data_features[location, :]
        i_center = np.mean(i_features[i], axis=0)

        # 径向的形变分析：
        dr = i_center - o_center
        oc_oi_distance = np.linalg.norm(dr, ord=2)
        dr = np.tile(dr / oc_oi_distance, (i_features[i].shape[0], 1))
        feauture_vectors = i_features[i] - np.tile(o_center, (i_features[i].shape[0], 1))
        dr_inner = np.sum(dr * feauture_vectors, axis=1)
        delta_dr = np.percentile(dr_inner, 98) - np.percentile(dr_inner, 2)
        # plt.figure()
        # plt.subplot(121)
        # plt.hist(dr_inner, bins=30)

        # 切向的形变分析：
        feauture_abs = np.sqrt(np.sum(feauture_vectors * feauture_vectors, axis=1))
        dt_inner = dr_inner / feauture_abs
        dt_inner[np.argwhere(dt_inner > 1).ravel()] = 1
        dt_inner_angle = np.arccos(dt_inner) / np.pi
        delta_dt_angle = np.percentile(dt_inner_angle, 98)
        # plt.subplot(122)
        # plt.hist(dt_inner_angle, bins=30)

        results[i, 0] = oc_oi_distance
        results[i, 1] = delta_dr/delta_dt_angle
        results[i, 2] = delta_dr
        results[i, 3] = delta_dt_angle

    plt.figure()
    plt.subplot(131)
    plt.scatter(results[:, 0], results[:, 1])

    # 一次拟合的结果
    model = LinearRegression()
    model.fit(results[:, 0].reshape(-1, 1), results[:, 1].reshape(-1, 1))
    plt.plot(results[:, 0], model.intercept_.ravel() + model.coef_.ravel() * results[:, 0],
             color='red',
             label='Linear($r^2$={:.2f})'.format(model.score(results[:, 0].reshape(-1, 1), results[:, 1].reshape(-1, 1))))
    # print(model.score(results[:, 0].reshape(-1, 1), results[:, 1].reshape(-1, 1)))
    # print(r2_score(results[:, 1], model.intercept_.ravel() + model.coef_.ravel() * results[:, 0]))

    # 二次拟合的结果
    poly_reg = PolynomialFeatures(degree=2)
    x_ploy = poly_reg.fit_transform(results[:, 0].reshape(-1, 1))
    lin_reg = LinearRegression()
    lin_reg.fit(x_ploy, results[:, 1].reshape(-1, 1))

    x = np.linspace(np.min(results[:, 0]), np.max(results[:, 0]), 100)
    xx = poly_reg.fit_transform(x.reshape(-1, 1))
    plt.plot(x, lin_reg.predict(xx), color='blue',
             label='Ploy($r^2=${:.2f})'.format(r2_score(results[:, 1], lin_reg.predict(x_ploy))))
    # print(r2_score(results[:, 1], lin_reg.predict(x_ploy)))

    plt.legend()

    plt.subplot(132)
    plt.scatter(results[:, 0], results[:, 2])

    plt.subplot(133)
    plt.scatter(results[:, 0], results[:, 3])

    plt.show()
