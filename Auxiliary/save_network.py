import torch
from Auxiliary.create_dir_file import name_transport
import os


def save_network(networks, criterion, exist_best_auc=False, **options):
    network_path = os.path.join(options['save_path']+'/network')
    if not os.path.exists(network_path):
        os.makedirs(network_path)
    name = name_transport(**options)

    weights = networks.state_dict()
    if exist_best_auc:
        net_name = '{}/{}_{}_best_auc.pth'.format(network_path, options['network'], name)
    else:
        net_name = '{}/{}_{}_{}.pth'.format(network_path, options['network'], name,
                                            str(options['epoch']+1)+'('+str(options['max_epoch'])+')')
    torch.save(weights, net_name)

    weights = criterion.state_dict()
    if exist_best_auc:
        cri_name = '{}/{}_{}_criterion_best_auc.pth'.format(network_path, options['network'], name)
    else:
        cri_name = '{}/{}_{}_{}_criterion.pth'.format(network_path, options['network'], name,
                                                      str(options['epoch']+1)+'('+str(options['max_epoch'])+')')
    torch.save(weights, cri_name)


def save_gan(generator, discriminator, generator_2=None, **options):
    network_path = os.path.join(options['save_path']+'/network')
    if not os.path.exists(network_path):
        os.makedirs(network_path)
    name = name_transport(**options)

    weights = generator.state_dict()
    net_name = '{}/Generator_{}_{}.pth'.format(network_path, name,
                                               str(options['epoch']+1)+'('+str(options['max_epoch'])+')')
    torch.save(weights, net_name)

    weights = discriminator.state_dict()
    net_name = '{}/Discriminator_{}_{}.pth'.format(network_path, name,
                                                   str(options['epoch']+1)+'('+str(options['max_epoch'])+')')
    torch.save(weights, net_name)

    if generator_2:
        weights = generator_2.state_dict()
        net_name = '{}/Generator2_{}_{}.pth'.format(network_path, name,
                                                    str(options['epoch'] + 1) + '(' + str(options['max_epoch']) + ')')
        torch.save(weights, net_name)
