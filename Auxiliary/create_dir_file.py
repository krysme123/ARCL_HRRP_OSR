import os
import datetime


def name_transport(**options):
    if options['loss'] == 'ARPLoss' and options['cs']:
        name = 'ARPL+CS'
    elif options['loss'] == 'ARPLoss':
        name = 'ARPL'
    elif options['loss'] == 'AKPFLoss' and options['cs++']:
        name = 'AKPF++'
    elif options['loss'] == 'AKPFLoss' and options['cs']:
        name = 'AKPF'
    elif options['loss'] == 'AKPFLoss':
        name = 'KPF'
    elif options['loss'] == 'GCPLoss':
        name = 'GCPL'
    elif options['loss'] == 'RPLoss':
        name = 'RPL'
    elif options['loss'] == 'RingLoss':
        name = 'Ring'
    elif options['loss'] == 'SLCPLoss':
        name = 'SLCPL'
    elif options['loss'] == 'SPLoss':
        name = 'SPL'
    elif options['loss'] == 'CenterLoss':
        name = 'Center'
    else:
        name = options['loss']
    return name


def create_dir_path(**options):         # 已修改
    # time = None
    # if torch.cuda.is_available():                   # windows 系统的文件夹名称里不允许带:，因此用 - 替代
    #     time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    # if torch.backends.mps.is_available():           # MacOS 系统的文件夹名称里允许带:
    #     time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ########################## 为了兼容 Windows 和 MacOS系统，把时间做如下设置
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    name = name_transport(**options)
    dir_path = os.path.join('Results_'+options['dataset']+'/'+options['network']+'_'+name+'/'+time)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def create_dir_path_tradtioinal(**options):
    dir_path = os.path.join('Results_'+options['dataset']+'/'+options['model']+'/')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path
