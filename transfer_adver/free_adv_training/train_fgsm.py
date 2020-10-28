# code adapted from https://github.com/locuslab/fast_adversarial

import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from tqdm import tqdm

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None, mu=None, std=None, device="cuda:0"):

    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)

    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def fast_free_adv_training(model, train_loader, batch_size = 128, epochs = 15, lr_schedule= 'cyclic', lr_min = 0, lr_max = 0.2, weight_decay= 5e-4, momentum= 0.9 , epsilon = 8, alpha = 10, delta_init ='random', seed = 0, early_stop = True, opt_level ='O0', loss_scale='1.0', master_weights=True, mean=None, std=None, device='cpu'):

    mu = torch.tensor(mean).view(3,1,1).to(device)
    std = torch.tensor(std).view(3,1,1).to(device)

    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    epsilon = (epsilon / 255.) / std
    alpha = (alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    model = model.to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
    amp_args = dict(opt_level=opt_level, loss_scale=loss_scale, verbosity=False)
    if opt_level == 'O2':
        amp_args['master_weights'] = master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    if delta_init == 'previous':
        delta = torch.zeros(batch_size, 3, 32, 32).to(device)

    lr_steps = epochs * len(train_loader)
    if lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()

    for epoch in tqdm(range(epochs)):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            if i == 0:
                first_batch = (X, y)
            if delta_init != 'previous':
                delta = torch.zeros_like(X).to(device)
            if delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)

            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        if early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt, mu=mu, std=std, device=device)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]

    if not early_stop:
        best_state_dict = model.state_dict()

    return model.float()
