import os
from Dataset.HRRP_datasets import HRRPOSRData310
from torch.utils.data import DataLoader
from Analysis.get_features_logits import get_features_logits
from Analysis.boundary_prototype import boundary_prototype
from Analysis.boundary_hypersphere import boundary_hypersphere
from Analysis.boundary_extreme_value import boundary_extreme_value
from Analysis.boundary_knn_lever_detection import boundary_knn_lever_detection
from Analysis.boundary_vector_detection import boundary_vector_detection
from Analysis.boundary_convexhull import boundary_tsne_convexhull
from Analysis.boundary_cos_angle import boundary_cos_angle


def openness_analysis(train_features, train_labels, train_logits,
                      test_features, test_labels, test_logits, test_pred_labels,
                      net, criterion, load_path, **options):
    print("现在开始 openness 测试，一共需要运行 {} 次！\n".format(len(options['unknown'])))

    for j in range(len(options['unknown'])):
        print("现在开始第 {} 次：\n".format(j))

        options['save_path'] = load_path + '/Openness_' + str(j)
        if not os.path.exists(options['save_path']):
            os.makedirs(options['save_path'])

        # ################################## 先加载数据 ###################################
        options['known'] = options['unknown'][:(j+1)]
        openness_data_set = HRRPOSRData310(**options)

        openness_loader = DataLoader(dataset=openness_data_set, batch_size=options['batch_size'], shuffle=False,
                                     num_workers=8, pin_memory=options['use_gpu'])

        openness_features, openness_labels, openness_logits, openness_pred_labels =\
            get_features_logits(net, criterion, openness_loader, **options)
        
        if options['boundary_type'] == 'Boundary_prototype':
            boundary_prototype(train_logits, train_labels, test_logits, test_labels, test_pred_labels,
                               openness_logits, openness_labels, openness_pred_labels, **options)
        elif options['boundary_type'] == 'Boundary_hypersphere':
            boundary_hypersphere(train_features, train_labels, test_features, test_labels, openness_features,
                                 openness_labels, **options)
        elif options['boundary_type'] == 'Boundary_hypersphere':
            boundary_extreme_value(train_features, train_labels, test_features, test_labels, openness_features,
                                   openness_labels, **options)
        elif options['boundary_type'] == 'Boundary_knn_lever':
            boundary_knn_lever_detection(train_features, train_labels, test_features, test_labels, openness_features,
                                         openness_labels, **options)
        elif options['boundary_type'] == 'Boundary_vector_detection':
            boundary_vector_detection(train_features, train_labels, test_features, test_labels, openness_features,
                                      openness_labels, **options)
        elif options['boundary_type'] == 'Boundary_tsne_convexhull':
            boundary_tsne_convexhull(train_features, train_labels, test_features, test_labels, openness_features,
                                     openness_labels, **options)
        elif options['boundary_type'] == 'Boundary_cos_angle':
            boundary_cos_angle(train_features, train_labels, test_features, test_labels, openness_features,
                               openness_labels, **options)
        else:
            raise ValueError("boundary_type的赋值出现错误！！！")
        