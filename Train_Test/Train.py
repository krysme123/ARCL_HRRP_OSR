import torch
from Train_Test.utils import AverageMeter
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as f


def train(net, criterion, optimizer, train_loader, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(options['device']), labels.to(options['device'])
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y = net(data, True)
            logic, loss = criterion(x, y, labels.long())
            loss.backward()
            optimizer.step()
        losses.update(loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss classifier: {:.6f} ({:.6f})"
                  .format(batch_idx + 1, len(train_loader), losses.val, losses.avg))
        loss_all += losses.avg
    return loss_all


# ############## 以下是 ARPL+CS 的训练过程，针对实数
def train_cs(net, net_discriminator, net_generator, criterion, criterion_discriminator, optimizer,
             optimizer_discriminator, optimizer_generator, train_loader, **options):
    print('train with confusing samples')
    losses, losses_generator, losses_discriminator = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    net_discriminator.train()
    net_generator.train()
    torch.cuda.empty_cache()

    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        gan_target = torch.FloatTensor(labels.size()).fill_(0).to(options['device'])
        data, labels = data.to(options['device']), labels.to(options['device'])
        data, labels = Variable(data), Variable(labels)

        if options['image_size'] == 0:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        else:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        noise = Variable(noise)

        fake = net_generator(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        target_variable = Variable(gan_target)
        optimizer_discriminator.zero_grad()
        output = net_discriminator(data)
        error_discriminator_real = criterion_discriminator(output, target_variable)
        # formula (15): log D(x_i)
        error_discriminator_real.backward()

        # train with fake
        target_variable = Variable(gan_target.fill_(fake_label))
        output = net_discriminator(fake.detach())
        error_discriminator_fake = criterion_discriminator(output, target_variable)
        # formula (15): log (1-D(G(z_i)))
        error_discriminator_fake.backward()
        error_discriminator = error_discriminator_real + error_discriminator_fake
        optimizer_discriminator.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizer_generator.zero_grad()
        # Original GAN loss
        target_variable = Variable(gan_target.fill_(real_label))
        output = net_discriminator(fake)
        error_generator = criterion_discriminator(output, target_variable)  # formula (18): log D(G(z_i))

        # minimize the true distribution
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        error_generator_fake = criterion.fake_loss(x).mean()  # formula (18): H(z_i, P)
        generator_loss = error_generator + options['beta'] * error_generator_fake
        generator_loss.backward()
        optimizer_generator.step()

        losses_generator.update(generator_loss.item(), labels.size(0))
        losses_discriminator.update(error_discriminator.item(), labels.size(0))

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        _, loss = criterion(x, y, labels)

        # KL divergence
        if options['image_size'] == 0:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        else:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        noise = Variable(noise)
        fake = net_generator(noise)
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        f_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + options['beta'] * f_loss_fake  # formula (19)
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})"
                  .format(batch_idx + 1, len(train_loader), losses.val, losses.avg, losses_generator.val,
                          losses_generator.avg, losses_discriminator.val, losses_discriminator.avg))
        loss_all += losses.avg
    return loss_all


# ############## 以下是 AKPF 的训练过程，针对实数
def train_cs_akpf(net, net_discriminator, net_generator, criterion, criterion_discriminator, optimizer,
                  optimizer_discriminator, optimizer_generator, train_loader, **options):
    print('train with confusing samples')
    losses, losses_generator, losses_discriminator = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    net_discriminator.train()
    net_generator.train()

    torch.cuda.empty_cache()
    loss_all, real_label, fake_label = 0, 1, 0

    r0 = torch.tensor(criterion.radius.item()).reshape(1).to(options['device'])
    for batch_idx, (data, labels) in enumerate(train_loader):
        o_center = criterion.points.mean(0, keepdim=True)
        d0 = (criterion.points - o_center).pow(2).mean(1).sum(0).detach().item()
        kappa = np.log(options['epoch'] + 3) * (options['gamma'] + d0 / r0)

        if options['epoch'] < 5:
            options['R_recording'].append(criterion.radius.cpu().detach().numpy())
            options['kR_recording'].append((kappa * criterion.radius).cpu().detach().numpy())

        gan_target = torch.FloatTensor(labels.size()).fill_(0).to(options['device'])

        data, labels = data.to(options['device']), labels.to(options['device'])
        data, labels = Variable(data), Variable(labels)

        if options['image_size'] == 0:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        else:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        noise = Variable(noise)
        fake = net_generator(noise)

        # (1) Update D network    #
        gan_target.fill_(real_label)                # train with real
        target_variable = Variable(gan_target)
        optimizer_discriminator.zero_grad()
        output = net_discriminator(data)
        error_discriminator_real = criterion_discriminator(output, target_variable)
        # formula (15): log D(x_i)
        error_discriminator_real.backward()

        # train with fake
        target_variable = Variable(gan_target.fill_(fake_label))
        output = net_discriminator(fake.detach())
        error_discriminator_fake = criterion_discriminator(output, target_variable)
        # formula (15): log (1-D(G(z_i)))
        error_discriminator_fake.backward()
        error_discriminator = error_discriminator_real + error_discriminator_fake
        optimizer_discriminator.step()

        # (2) Update G network    #
        optimizer_generator.zero_grad()
        # Original GAN loss
        target_variable = Variable(gan_target.fill_(real_label))
        output = net_discriminator(fake)
        error_generator = criterion_discriminator(output, target_variable)  # formula (18): log D(G(z_i))

        # my H_1
        fake_features, _ = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        _dis_known = (fake_features - o_center).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).to(options['device'])
        j_zi = criterion.margin_loss(_dis_known, kappa * r0, target)

        generator_loss = error_generator + options['alpha'] * j_zi
        generator_loss.backward()
        optimizer_generator.step()

        losses_generator.update(generator_loss.item(), labels.size(0))
        losses_discriminator.update(error_discriminator.item(), labels.size(0))

        # (3) Update classifier   #
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        _, loss = criterion(x, y, labels)

        # KL divergence
        if options['image_size'] == 0:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        else:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns'])
                     .normal_(0, 1).to(options['device']))

        noise = Variable(noise)
        fake = net_generator(noise)
        fake_features, _ = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))

        _dis_known = (fake_features - o_center).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).to(options['device'])
        j0_zi = criterion.margin_loss(_dis_known, kappa * criterion.radius, target)

        total_loss = loss + options['beta'] * j0_zi
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) \t G {:.3f} ({:.3f}) J0 {:.3f}\t"
                  " D {:.3f} ({:.3f})"
                  .format(batch_idx + 1, len(train_loader), losses.val, losses.avg,
                          losses_generator.val, losses_generator.avg, j0_zi.item(),
                          losses_discriminator.val, losses_discriminator.avg))
            print("R:{:.3f}".format(criterion.radius.item()))
        loss_all += losses.avg
    print()
    return loss_all


# ############## 以下是 AKPF++ 的训练过程，针对实数
def train_cs_akpf_plus(net, net_generator_2, criterion, optimizer,
                       optimizer_generator_2, train_loader, **options):
    print('train with confusing samples plus')
    losses = AverageMeter()
    losses_generator_2 = AverageMeter()

    net.train()
    net_generator_2.train()
    torch.cuda.empty_cache()

    loss_all, real_label, fake_label = 0, 1, 0

    r0 = torch.tensor(criterion.radius.item()).reshape(1).to(options['device'])
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(options['device']), labels.to(options['device'])
        data, labels = Variable(data), Variable(labels)

        o_center = criterion.points.mean(0)
        d0 = (criterion.points - o_center).pow(2).mean(1).sum(0).detach().item()
        kappa = np.log(options['epoch'] + 3) * (options['gamma'] + d0 / r0)

        if options['epoch'] < 5:
            options['R_recording'].append(criterion.radius.cpu().detach().numpy())
            options['kR_recording'].append((kappa * criterion.radius).cpu().detach().numpy())

        if options['image_size'] == 0:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        else:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        noise = Variable(noise)
        fake_2 = net_generator_2(noise)

        # (1) Update G_2 network  #
        delta_x = torch.randn((data.shape[0], len(o_center))).to(options['device'])
        delta_x = delta_x * (criterion.points-o_center).pow(2).mean(1).mean() / (1 + 3 * np.sqrt(2/len(o_center)))
        optimizer_generator_2.zero_grad()

        # Original GAN loss
        fake_features, _ = net(fake_2, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        error_generator_2 = f.mse_loss(o_center.repeat(fake_features.shape[0], 1) + delta_x, fake_features)

        generator_loss_2 = error_generator_2
        generator_loss_2.backward()
        optimizer_generator_2.step()

        losses_generator_2.update(generator_loss_2.item(), labels.size(0))

        # (2) Update classifier   #
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        _, loss = criterion(x, y, labels)

        # KL divergence
        if options['image_size'] == 0:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        else:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        noise = Variable(noise)
        fake = net_generator_2(noise)
        fake_features, _ = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))

        # my J_2
        _dis_known = (fake_features - o_center).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).to(options['device'])
        j2_zi = criterion.margin_loss(_dis_known, kappa * criterion.radius, target)

        total_loss = loss + options['beta'] * j2_zi
        total_loss.backward()
        optimizer.step()
        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) \t"
                  "G_2 {:.3f} ({:.3f}) J2 {:.4f}"
                  .format(batch_idx + 1, len(train_loader), losses.val, losses.avg,
                          losses_generator_2.val, losses_generator_2.avg, j2_zi.item()))
            print("R:{:.4f}".format(criterion.radius.item()))
        loss_all += losses.avg
    print()
    return loss_all


# ############## 以下是 ARPL+CS 的训练过程，针对复数
def train_cs_complex(net, net_discriminator, net_generator, criterion, criterion_discriminator, optimizer,
                     optimizer_discriminator, optimizer_generator, train_loader, **options):
    print('train with confusing samples')
    losses, losses_generator, losses_discriminator = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    net_discriminator.train()
    net_generator.train()
    torch.cuda.empty_cache()

    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        gan_target = torch.FloatTensor(labels.size()).fill_(0).to(options['device'])
        data, labels = data.to(options['device']), labels.to(options['device'])
        data, labels = Variable(data), Variable(labels)

        if options['image_size'] == 0:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        else:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        noise = Variable(noise)

        fake = net_generator(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        target_variable = Variable(gan_target)
        optimizer_discriminator.zero_grad()
        output = net_discriminator(data)
        error_discriminator_real = criterion_discriminator(output, target_variable)
        # formula (15): log D(x_i)
        error_discriminator_real.backward()

        # train with fake
        target_variable = Variable(gan_target.fill_(fake_label))
        output = net_discriminator(fake.detach())
        error_discriminator_fake = criterion_discriminator(output, target_variable)
        # formula (15): log (1-D(G(z_i)))
        error_discriminator_fake.backward()
        error_discriminator = error_discriminator_real + error_discriminator_fake
        optimizer_discriminator.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizer_generator.zero_grad()
        # Original GAN loss
        target_variable = Variable(gan_target.fill_(real_label))
        output = net_discriminator(fake)
        error_generator = criterion_discriminator(output, target_variable)  # formula (18): log D(G(z_i))

        # minimize the true distribution
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        error_generator_fake = criterion.fake_loss(x).mean()  # formula (18): H(z_i, P)
        generator_loss = error_generator + options['beta'] * error_generator_fake
        generator_loss.backward()
        optimizer_generator.step()

        losses_generator.update(generator_loss.item(), labels.size(0))
        losses_discriminator.update(error_discriminator.item(), labels.size(0))

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        _, loss = criterion(x, y, labels)

        # KL divergence
        if options['image_size'] == 0:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        else:
            noise = (torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns'])
                     .normal_(0, 1).to(options['device']))
        noise = Variable(noise)
        fake = net_generator(noise)
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        f_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + options['beta'] * f_loss_fake  # formula (19)
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})"
                  .format(batch_idx + 1, len(train_loader), losses.val, losses.avg, losses_generator.val,
                          losses_generator.avg, losses_discriminator.val, losses_discriminator.avg))
        loss_all += losses.avg
    return loss_all