
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from Auxiliary.create_logfile import create_iir_logfile


def compute_iir(features, labels, **options):
    c_proto = np.array([features[np.argwhere(labels == p).ravel()].mean(0) for p in range(options['num_classes'])])
    distance_features, intra_class_distance = {}, 0
    for p in range(options['num_classes']):
        distance_features[str(p)] = pairwise_distances(
            features[np.argwhere(labels == p).ravel()], c_proto[p, :].reshape(1, -1),
            metric="euclidean", n_jobs=-1).ravel()
        intra_class_distance += distance_features[str(p)].mean(0)
    inter_class_distance = np.sum(pairwise_distances(c_proto, c_proto, metric='euclidean', n_jobs=-1))/2
    iir = intra_class_distance / inter_class_distance

    print('IIR是：{:.3f}'.format(iir))
    options['IIR'] = iir
    create_iir_logfile(**options)
