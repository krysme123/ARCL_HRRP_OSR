import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os


def plot_confusion_matrix(y_true, y_pred, y_classes, x_classes, threshold=None,
                          rotation=False, **options):
    """
    :param y_true:
    :param y_pred:
    :param y_classes:       the row name of confusion matrix;
    :param x_classes:       the column name of confusion matrix;
    :param save_path:       the name of figure, already include save_path;
    :param threshold:
    :param rotation:
    :return:                nothing;
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
    cm = cm_normalized[:, 0:(options['num_classes'] + 1)]

    plt.figure(figsize=(12, 8), dpi=100)

    x, y = np.meshgrid(np.arange(cm.shape[1]), np.arange(cm.shape[0]))

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.005:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm[:, 0:len(x_classes)], cmap='viridis')
    plt.colorbar()
    xlocations = np.array(range(len(x_classes)))
    if rotation:
        plt.xticks(xlocations, x_classes, rotation=45)
    else:
        plt.xticks(xlocations, x_classes)
    ylocations = np.array(range(len(y_classes)))
    plt.yticks(ylocations, y_classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    tick_marks = np.array(range(len(y_classes))) + 0.5
    plt.gca().set_yticks(tick_marks, minor=True)
    tick_marks = np.array(range(len(x_classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)

    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    if options['save_path']:
        if threshold:
            plt.savefig(options['save_path'] + '/' + options['boundary_type'] + '_cm_'
                        + str(threshold) + '.pdf')
        else:
            plt.savefig(options['save_path'] + '/' + options['boundary_type'] + '_cm.pdf')
    plt.close()
    # plt.show()
