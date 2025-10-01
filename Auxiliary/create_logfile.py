def create_logfile(**options):
    file = open(options['save_path'] + '/logfile.txt', 'w')
    text = (
            'Path: {}\n'.format(options['save_path']) +
            'dataset: {}\n'.format(options['dataset']) +
            'device: {}\n'.format(options['device']) +
            'known classes: {}\n'.format(options['known']) +
            'unknown classes: {}\n'.format(options['unknown']) +
            'network: {}\n'.format(options['network']) +
            'feature dimension: {}\n'.format(options['feature_dim']) +
            'image size: {}\n'.format(options['image_size']) +
            'batch size: {}\n'.format(options['batch_size']) +
            'max epoch: {}\n'.format(options['max_epoch']) +
            'complex: {}\n'.format(options['complex']) +
            'number classes: {}\n'.format(options['num_classes']) +
            'loss function: {}\n'.format(options['loss']) +
            'cs: {}\n'.format(options['cs']) +
            'cs++: {}\n'.format(options['cs++']) +
            'learning rate: {}\n'.format(options['lr']) +
            'optimizer: {}\n'.format(options['optimizer_']) +
            'scheduler type: {}\n'.format(options['scheduler']) +
            'label smoothing: {}\n'.format(options['label_smoothing']) +
            'final acc: {}\n'.format(options['acc']) +
            'final auc: {}\n'.format(options['auc']) +
            'final oscr: {}\n'.format(options['oscr']) +
            'best auc: {}\n'.format(options['best_auc']) +
            'best auc epoch: {}\n'.format(options['best_auc_epoch'])
    )
    file.write(text)
    file.close()


def create_traditional_logfile(**options):
    file = open(options['save_path'] + '/logfile.txt', 'w')
    text = (
            'Path: {}\n'.format(options['save_path']) +
            'dataset: {}\n'.format(options['dataset']) +
            'device: {}\n'.format(options['device']) +
            'known classes: {}\n'.format(options['known']) +
            'unknown classes: {}\n'.format(options['unknown']) +
            'complex: {}\n'.format(options['complex']) +
            'number classes: {}\n'.format(options['num_classes']) +
            'svm csr acc: {}\n'.format(options['acc']) +
            'model auc: {}\n'.format(options['auc']) +
            'svm + model macro-f1: {}\n'.format(options['f1'])
    )
    file.write(text)
    file.close()


def create_boundary_analysis_logfile(**options):
    file = open(options['save_path'] + '/{}_analysis.txt'.format(options['boundary_type']), 'w')
    text = (
            'Path: {}\n'.format(options['save_path']) +
            'boundary type: {}\n'.format(options['boundary_type']) +
            'dataset: {}\n'.format(options['dataset']) +
            'device: {}\n'.format(options['device']) +
            'known classes: {}\n'.format(options['known']) +
            'unknown classes: {}\n'.format(options['unknown']) +
            'complex: {}\n'.format(options['complex']) +
            'number classes: {}\n'.format(options['num_classes']) +
            'auc: {}\n'.format(options['auc']) +
            'macro-f1: {}\n'.format(options['macro_f1'])
    )
    file.write(text)
    file.close()


def create_iir_logfile(**options):
    file = open(options['save_path'] + '/IIR_logfile.txt', 'w')
    text = (
            'IIR: {}\n'.format(options['IIR'])
    )
    file.write(text)
    file.close()
