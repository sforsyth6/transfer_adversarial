from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision.datasets import FashionMNIST
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import time
import argparse
import random
from advertorch.attacks import LinfPGDAttack
from masked_pgd import MaskedPGDAttack
import time
from collections import defaultdict
from CustomImageNetTest import CustomImageNetTest
from free_adv_training.train_fgsm import *
from torch.autograd import Variable
from OneCycle import *

from torch.utils.data import Dataset, DataLoader

class SplitDataset(Dataset):
    """Abstraction of dataset to allow for splits easier"""

    def __init__(self, indicies, dataset, class_mappings):
        """
        Args:
            indicies (list): indicies for all points within dataset
            dataset (object): dataset of all data
            class_mappings (dict): how classes should be mapped per model
                on a sample.
        """
        self.indicies = indicies
        self.dataset = dataset
        self.class_mappings = class_mappings

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, idx):
        new_idx = int(self.indicies[idx])
        return (self.dataset[new_idx][0], self.class_mappings[int(self.dataset[new_idx][1])])

        return self.dataset[new_idx]

def accuracy(predictions, true_y):

    correct = 0

    for i in range(len(predictions)):
        if int(torch.argmax(predictions[i])) == int(true_y[i]):
            correct += 1

    accuracy = correct / len(true_y)
    return accuracy


def accuracy_n(predictions, true_y, n):

    correct = 0

    for i in range(len(predictions)):
        found = False
        for j in range(1,n+1):
            if int(torch.argsort(predictions[i])[-j]) == int(true_y[i]):
                found = True
        if found:
            correct += 1

    accuracy = correct / len(true_y)
    return accuracy

def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)

def normalize_imgs(imgs, mean, std):
    nrm_fxm = transforms.Normalize(mean,std)
    nrm_imgs = []
    for img in imgs:
        nrm_imgs.append(nrm_fxm(img))

    return torch.stack(nrm_imgs)


def experiment(num_shared_classes, percent_shared_data, n_epochs=200,batch_size=128, eps=.3, adv_steps=1000, learning_rate=.0004, gpu_num=1,adv_training="none",task="CIFAR100", masked=False, savemodel=False, download_data=False):
    print("epochs,batch_size,eps,adv_steps,learning_rate,task")
    print(n_epochs,batch_size,eps,adv_steps,learning_rate,task)

    cuda = torch.cuda.is_available()


    if task.upper() == "CIFAR100":
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        train_data = CIFAR100("data/",transform=transform_train, download=download_data)
        test_data = CIFAR100("data/", train=False, transform=transform_test, download=download_data)
    elif task.upper() == "FASHIONMNIST":

        mean = (0.2860)
        std = (0.3530)

        transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                        transforms.ToTensor()
                             ])

        train_data = FashionMNIST('data/fashionmnist',transform=transform, train=True, download=download_data)
        test_data = FashionMNIST('data/fashionmnist', transform=transform, train=False, download=download_data)

    elif task.upper() == "IMAGENET":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_data = ImageFolder(args.data + '/train', transform=transform_train)

        #################################### change to ImageFolder instead of CustomImageFolderTest
        test_data = ImageFolder(args.data + '/val/', transform=transform_test)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        transform_test = transforms.Compose(
                [transforms.ToTensor(),transforms.Normalize(mean, std)])

        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
        train_data = CIFAR10("data/",transform=transform_train,download=download_data)
        test_data = CIFAR10("data/", train=False, transform=transform_test,download=download_data)

    ######################### Change all_classes so it's not looping over 1.2 million images
    if task.upper() == "IMAGENET":
        all_classes = range(len(train_data.classes))
    else:
        all_classes = set([x[1] for x in train_data])
    shared_classes = random.sample(all_classes, num_shared_classes)
    split_classes = [c for c in all_classes if c not in shared_classes] # get classes not shared


    if len(split_classes) % 2 == 1: # if we have an odd #, randomly remove one so that number of classes will be the same for each model
        split_classes.pop(random.randint(0, len(split_classes) - 1))

    model1_split = random.sample(split_classes, len(split_classes) // 2)
    model2_split = [c for c in split_classes if c not in model1_split]

    model1_classes = model1_split
    model2_classes = model2_split

    model1_classes.sort()
    model2_classes.sort()

    # DEBUG:
    print("shared classes: {}".format(shared_classes))
    print("model1 classes: {}".format(model1_classes))
    print("model2 classes: {}".format(model2_classes))

    # split
    ############################## Change the way classes_by_index is generated to avoid looping over dataset
    if task.upper() == "IMAGENET":
        classes_by_index = []
        for idx, label in enumerate(train_data.classes):
            label_size = len([x for x in os.listdir(os.path.join(args.data, "train", label)) if ".JPEG" in x])
            for i in range(label_size):
                classes_by_index.append(idx)
        classes_by_index = np.array(classes_by_index)
    else:
        classes_by_index = np.array([train_data[index][1] for index in range(len(train_data))])

    model1_train_indicies = np.array([])
    model2_train_indicies = np.array([])

    # cut back on percentage of data
    if percent_shared_data < 100:
        d = defaultdict(list)
        for i in range(len(classes_by_index)):
            d[classes_by_index[i]].append(i)

        model1_train_indicies = np.array([])
        model2_train_indicies = np.array([])

        for key in d.keys():
            if key in model1_classes:
                np.random.shuffle(d[key])
                new_len = len(d[key]) // 2
                model1_train_indicies = np.concatenate((d[key][:new_len], model1_train_indicies))

            if key in model2_classes:
                np.random.shuffle(d[key])
                new_len = len(d[key]) // 2
                model2_train_indicies = np.concatenate((d[key][:new_len], model2_train_indicies))
    else:
        model1_train_indicies = np.argwhere(np.isin(classes_by_index, model1_classes) == True)
        model2_train_indicies = np.argwhere(np.isin(classes_by_index, model2_classes) == True)

    # split up shared data

    # divide shared data
    if percent_shared_data < 100:
        # split based on shared classes
        for curr_class in shared_classes:
            temp_x = np.argwhere(classes_by_index == curr_class).reshape(-1)

            size_shared = int(len(temp_x) * (percent_shared_data / 100))

            np.random.shuffle(temp_x)

            # joint shared data
            shared_indicies = temp_x[:size_shared]

            # add the joint data to datasets
            model1_train_indicies = np.concatenate((model1_train_indicies.reshape(-1), shared_indicies))
            model2_train_indicies = np.concatenate((model2_train_indicies.reshape(-1), shared_indicies))

            # disjoint shared class data
            point = (len(temp_x) - size_shared) // 2

            model1_train_indicies = np.concatenate((model1_train_indicies.reshape(-1), temp_x[size_shared:size_shared+point]))
            model2_train_indicies = np.concatenate((model2_train_indicies.reshape(-1), temp_x[size_shared+point:]))

    else:
        shared_data_indicies = np.argwhere(np.isin(classes_by_index, shared_classes) == True).reshape(-1)

        model1_train_indicies = np.concatenate((model1_train_indicies.reshape(-1), shared_data_indicies))
        model2_train_indicies = np.concatenate((model2_train_indicies.reshape(-1), shared_data_indicies))

    # create class mappings

    model1_class_mapping = {}
    model2_class_mapping = {}

    model1_classes_inc = 0
    # go through model1 and assign unique classes to incrimental int starting at 0
    for c in (shared_classes + model1_classes):
        # if it doesn't exist assign
        if c not in model1_class_mapping.keys():
            model1_class_mapping[c] = model1_classes_inc
            model1_classes_inc += 1

    model2_classes_inc = 0
    # go through model2 and assign unique classes to incrimental int starting at 0
    for c in (shared_classes + model2_classes):
        # if it doesn't exist in model2 OR in model1, assign it
        if c not in model2_class_mapping.keys() and c not in model1_class_mapping.keys():
            model2_class_mapping[c] = model2_classes_inc
            model2_classes_inc += 1
        if c in model1_class_mapping.keys():
            model2_class_mapping[c] = model1_class_mapping[c]

    model1_classes_len = len(model1_class_mapping.keys())
    model2_classes_len = len(model2_class_mapping.keys())

    if task.upper() == "CIFAR100":

        model1 = models.wide_resnet50_2()
        model2 = models.wide_resnet50_2()
        #
        model1.fc = nn.Linear(2048, model1_classes_len)
        model2.fc = nn.Linear(2048, model2_classes_len)

    elif task.upper() == "IMAGENET":
        if args.model == 'resnet18':
            model1 = models.resnet18()
            model2 = models.resnet18()

            model1.fc = nn.Linear(512, model1_classes_len)
            model2.fc = nn.Linear(512, model2_classes_len)
        
        elif args.model == 'resnet50':
            model1 = models.resnet50()
            model2 = models.resnet50()

            model1.fc = nn.Linear(2048, model1_classes_len)
            model2.fc = nn.Linear(2048, model2_classes_len)

        elif args.model == 'resnet50_2':
            model1 = models.wide_resnet50_2()
            model2 = models.wide_resnet50_2()

            model1.fc = nn.Linear(2048, model1_classes_len)
            model2.fc = nn.Linear(2048, model2_classes_len)


    elif task.upper() == "FASHIONMNIST":
        model1 = models.resnet18()
        model2 = models.resnet18()


        model1.fc = nn.Linear(512, model1_classes_len)
        model2.fc = nn.Linear(512, model2_classes_len)
    else:
        # Get model (using ResNet50 for now)
        model1 = models.resnet50()
        model2 = models.resnet50()

        model1.fc = nn.Linear(2048, model1_classes_len)
        model2.fc = nn.Linear(2048, model2_classes_len)


    ################ Changed way cuda device was called and add DataParallel for the models

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        model1 = torch.nn.DataParallel(model1, device_ids=range(torch.cuda.device_count()))
        model2 = torch.nn.DataParallel(model2, device_ids=range(torch.cuda.device_count()))

    # Model Training

    model1 = model1.to(device)
    model2 = model2.to(device)

    if task.upper() != "IMAGENET":
        criterion1 = nn.CrossEntropyLoss().cuda()
        optimizer1 = optim.AdamW(model1.parameters(), lr=learning_rate)
        scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1,milestones=[60, 120, 160], gamma=.2) #learning rate decay


        criterion2 = nn.CrossEntropyLoss().cuda()
        optimizer2 = optim.AdamW(model2.parameters(), lr=learning_rate)
        scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2,milestones=[60, 120, 160], gamma=.2) #learning rate decay

    else:

        onecycle1 = OneCycle(int(len(model1_train_indicies) * n_epochs / batch_size), 0.8, prcnt=(n_epochs - 82) * 100/n_epochs, momentum_vals=(0.95, 0.8))
        onecycle2 = OneCycle(int(len(model2_train_indicies) * n_epochs /batch_size), 0.8, prcnt=(n_epochs - 82) * 100/n_epochs, momentum_vals=(0.95, 0.8))
        
        criterion1 = nn.CrossEntropyLoss()
        optimizer1 =  optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.95, weight_decay=1e-4)

        criterion2 = nn.CrossEntropyLoss()
        optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.95, weight_decay=1e-4)


    # make datasets
    model1_train_dataset = SplitDataset(model1_train_indicies, train_data, model1_class_mapping)
    model2_train_dataset = SplitDataset(model2_train_indicies, train_data, model2_class_mapping)

    # create trainloader 1
    trainloader_1 = torch.utils.data.DataLoader(model1_train_dataset, batch_size=batch_size,
                                              shuffle=True, pin_memory=True, num_workers=args.workers)
    # create trainloader 2
    trainloader_2 = torch.utils.data.DataLoader(model2_train_dataset, batch_size=batch_size,
                                              shuffle=True, pin_memory=True, num_workers=args.workers)

    # get test sets ready
    ############################## Change the way test_classes_by_index is generated to avoid looping over dataset
    if task.upper() == "IMAGENET":
        test_classes_by_index = []
        for idx, label in enumerate(test_data.classes):
            label_size = len([x for x in os.listdir(os.path.join(args.data, "val", label)) if ".JPEG" in x])
            for i in range(label_size):
                test_classes_by_index.append(idx)
        test_classes_by_index = np.array(test_classes_by_index)
    else:
        test_classes_by_index = np.array([test_data[index][1] for index in range(len(test_data))])

    model1_test_indicies = np.array([])
    model2_test_indicies = np.array([])
    shared_test_indicies = np.array([])

    model1_test_indicies = np.argwhere(np.isin(test_classes_by_index, model1_classes) == True)
    model2_test_indicies = np.argwhere(np.isin(test_classes_by_index, model2_classes) == True)
    shared_test_indicies = np.argwhere(np.isin(test_classes_by_index, shared_classes) == True)

    model1_test_indicies = np.concatenate([model1_test_indicies, shared_test_indicies])
    model2_test_indicies = np.concatenate([model2_test_indicies, shared_test_indicies])

    model1_test_dataset = SplitDataset(model1_test_indicies, test_data, model1_class_mapping)
    model2_test_dataset = SplitDataset(model2_test_indicies, test_data, model2_class_mapping)
    shared_test_dataset = SplitDataset(shared_test_indicies, test_data, model2_class_mapping) # does not matter which mapping


    # # dataloaders
    testloader_1 = torch.utils.data.DataLoader(model1_test_dataset, batch_size=len(model1_test_dataset),
                                              shuffle=True, pin_memory=True, num_workers=args.workers)

    testloader_2 = torch.utils.data.DataLoader(model2_test_dataset, batch_size=len(model2_test_dataset),
                                              shuffle=True, pin_memory=True, num_workers=args.workers)

    testloader_shared = torch.utils.data.DataLoader(shared_test_dataset, batch_size=batch_size,
                                              shuffle=True, pin_memory=True, num_workers=args.workers)

    # shared_x_test = []
    # shared_y_test = []
    # for i in range(len(shared_test_dataset)):
    #     data = shared_test_dataset[i]
    #     shared_x_test.append(data[0])
    #     shared_y_test.append(data[1])
    #
    # shared_x_test = torch.stack(shared_x_test)
    # shared_y_test = torch.tensor(shared_y_test)

    if adv_training == "fast_free":

        model1 = fast_free_adv_training(model1, trainloader_1, epochs=n_epochs, mean=mean, std=std, device=device, early_stop=False)
        print('Finished Training model1 adversarially')
        model2 = fast_free_adv_training(model2, trainloader_2, epochs=n_epochs, mean=mean, std=std, device=device, early_stop=False)
        print('Finished Training model2 adversarially')

    # Adversarial Training for Free! (adapted from https://github.com/mahyarnajibi/FreeAdversarialTraining)
    elif adv_training == "free":
        # mean = torch.tensor(mean).expand(3,32,32).to(device)
        # std = torch.tensor(std).expand(3,32,32).to(device)

        global_noise_data = torch.zeros([batch_size, 3, 32, 32]).to(device)

        # train model 1
        for epoch in tqdm(range(n_epochs),desc="Epoch"):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader_1, 0):
                #if cuda: ################################################################################check  here if broke
                    #data = tuple(d.cuda() for d in data)

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # steps for adv training
                for j in range(4):
                    noise_batch = Variable(global_noise_data[0:inputs.size(0)], requires_grad=True).to(device)

                    in1 = inputs + noise_batch
                    in1.clamp_(0, 1.0)
                    in1 = normalize_imgs(in1,mean,std)
                    output = model1(in1)
                    loss = criterion1(output, labels)
                    # compute gradient and do SGD step
                    optimizer1.zero_grad()
                    loss.backward()

                    # Update the noise for the next iteration
                    pert = fgsm(noise_batch.grad, 4)
                    global_noise_data[0:inputs.size(0)] += pert.data
                    global_noise_data.clamp_(-4.0, 4.0)

                    optimizer1.step()

        global_noise_data = torch.zeros([batch_size, 3, 32, 32]).to(device)
        # train model 2
        for epoch in tqdm(range(n_epochs),desc="Epoch"):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader_2, 0):
                #if cuda: ################################################################################check  here if broke
                #    data = tuple(d.cuda() for d in data)

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # steps for adv training
                for j in range(4):
                    noise_batch = Variable(global_noise_data[0:inputs.size(0)], requires_grad=True).to(device)

                    in1 = inputs + noise_batch
                    in1.clamp_(0, 1.0)
                    in1 = normalize_imgs(in1,mean,std)
                    output = model2(in1)
                    loss = criterion2(output, labels)
                    # compute gradient and do SGD step
                    optimizer2.zero_grad()
                    loss.backward()

                    # Update the noise for the next iteration
                    pert = fgsm(noise_batch.grad, 4)
                    global_noise_data[0:inputs.size(0)] += pert.data
                    global_noise_data.clamp_(-4.0, 4.0)

                    optimizer2.step()


    else:
        # train model 1
        for epoch in tqdm(range(n_epochs),desc="Epoch"):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader_1, 0):
                #if cuda: ################################################################################check  here if broke
                #    data = tuple(d.cuda() for d in data)

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # one cycle policy
                if task.upper() == "IMAGENET":
                    lr, mom = onecycle1.calc()
                    update_lr(optimizer1, lr)
                    update_mom(optimizer1, mom)

                # zero the parameter gradients
                optimizer1.zero_grad()

                # forward + backward + optimize

                outputs = model1(inputs)
                loss = criterion1(outputs, labels)
                loss.backward()
                optimizer1.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i  + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training model1')

        # train model 2
        for epoch in tqdm(range(n_epochs),desc="Epoch"):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader_2, 0):
                #if cuda: ################################################################################check  here if broke
                #    data = tuple(d.cuda() for d in data)

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # one cycle policy
                if task.upper() == "IMAGENET":
                    lr, mom = onecycle2.calc()
                    update_lr(optimizer2, lr)
                    update_mom(optimizer2, mom)

                # zero the parameter gradients
                optimizer2.zero_grad()

                # forward + backward + optimize
                outputs = model2(inputs)
                loss = criterion2(outputs, labels)
                loss.backward()
                optimizer2.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training model2')

    model1.eval()

    print("Running attack...")

    if masked:
        adversary = MaskedPGDAttack(
            model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=adv_steps, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False, device=device)
    else:
        adversary = LinfPGDAttack(
            model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=adv_steps, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)

    shared_x_test = []
    shared_y_test = []

    adv_untargeted = []

    for data in testloader_shared:
        x_test = data[0].float().to(device)
        y_test = data[1].to(device)

        adv_x = adversary.perturb(x_test, y_test)

        adv_untargeted.append(adv_x)
        shared_x_test.append(x_test)
        shared_y_test.append(y_test)

    adv_untargeted = torch.cat(adv_untargeted)
    shared_x_test = torch.cat(shared_x_test)
    shared_y_test = torch.cat(shared_y_test)

    # adv_untargeted = adversary.perturb(shared_x_test, shared_y_test)

    timestr = time.strftime("%Y%m%d_%H%M%S")

    model1 = model1.to("cpu")
    model2 = model2.to("cpu")

    model1_name = args.save_model_dir + '/{}_{}_{}_model1_{}.pickle'.format(task,num_shared_classes, percent_shared_data,timestr)
    model2_name = args.save_model_dir + '/{}_{}_{}_model2_{}.pickle'.format(task,num_shared_classes, percent_shared_data,timestr)
    adv_name = args.save_model_dir + '/{}_{}_{}_adv_{}.pickle'.format(task,num_shared_classes, percent_shared_data,timestr)

    if savemodel:
        print("saving models at", timestr)
        torch.save(model1, model1_name)
        torch.save(model2, model2_name)
        torch.save(adversary, adv_name)


    model1_x_test = []
    model1_y_test = []
    for i in range(len(model1_test_dataset)):
        data = model1_test_dataset[i]
        model1_x_test.append(data[0])
        model1_y_test.append(data[1])

    model1_x_test = torch.stack(model1_x_test)
    model1_y_test = torch.tensor(model1_y_test)

    model2_x_test = []
    model2_y_test = []
    for i in range(len(model2_test_dataset)):
        data = model2_test_dataset[i]
        model2_x_test.append(data[0])
        model2_y_test.append(data[1])

    model1 = model1.to(device)
    model2 = model2.to(device)

    model2_x_test = torch.stack(model2_x_test)
    model2_y_test = torch.tensor(model2_y_test)

    model1_x_test = model1_x_test.to(device)
    model2_x_test = model2_x_test.to(device)

    #  Eval
    with torch.no_grad():
        model1.eval()
        model2.eval()

        # model1 outputs

        output1 = model1(model1_x_test)
        shared_output1 = model1(shared_x_test)
        adv_output1 = model1(adv_untargeted)

        # model2 outputs
        output2 = model2(model2_x_test)
        shared_output2 = model2(shared_x_test)
        adv_output2 = model2(adv_untargeted)

        if task.upper() == "CIFAR100":

            # model 1

            print("model1_acc:", accuracy(output1,model1_y_test))

            print("model1_acc_5:", accuracy_n(output1,model1_y_test,5))

            print("model1_acc_shared:", accuracy(shared_output1,shared_y_test))
            print("model1_acc_5_shared:", accuracy_n(shared_output1,shared_y_test,5))

            print("model1_adv_acc_shared:", accuracy(adv_output1,shared_y_test))
            print("model1_adv_acc_5_shared:", accuracy_n(adv_output1,shared_y_test,5))

            print()

            # model 2

            print("model2_acc:", accuracy(output2,model2_y_test))
            print("model2_acc_5:", accuracy_n(output2,model2_y_test,5))

            print("model2_acc_shared:", accuracy(shared_output2,shared_y_test))
            print("model2_acc_5_shared:", accuracy_n(shared_output2,shared_y_test,5))

            print("model2_adv_acc_shared:", accuracy(adv_output2,shared_y_test))
            print("model2_adv_acc_5_shared:", accuracy_n(adv_output2,shared_y_test,5))

        else:
             # model 1

            print("model1_acc:", accuracy(output1,model1_y_test))

            print("model1_acc_shared:", accuracy(shared_output1,shared_y_test))

            print("model1_adv_acc_shared:", accuracy(adv_output1,shared_y_test))
            print()

            # model 2

            print("model2_acc:", accuracy(output2,model2_y_test))

            print("model2_acc_shared:", accuracy(shared_output2,shared_y_test))

            print("model2_adv_acc_shared:", accuracy(adv_output2,shared_y_test))



import argparse

parser = argparse.ArgumentParser(description='Experiment')

parser.add_argument('--task', default="CIFAR100")
parser.add_argument('--epochs', default=200)
parser.add_argument('--gpu', default=0)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--masked', action='store_true')
parser.add_argument('--savemodels', action='store_true', default=False)
parser.add_argument('--advtrain', default="none")
parser.add_argument('--loops', default=1)
parser.add_argument('--adv_steps', default=1000)
parser.add_argument('--download_data', action='store_true', default=False)
parser.add_argument('--shared_classes', default=None)
parser.add_argument('--shared_percent', default=None)
parser.add_argument('--data', default="/data/")
parser.add_argument('--save-model-dir', default='/results/')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--model', default='resnet50_2', help="choose model: resnet18, resnet50, resnet50_2")

args = parser.parse_args()

if args.shared_classes != None and args.shared_percent != None:
    loop_class_list = [int(args.shared_classes)]
    loop_percent_list = [int(args.shared_percent)]
elif args.task.upper() == "CIFAR100":
    loop_class_list = [2,25,50,75,100]
    loop_percent_list = [0,25,50,75,100]
elif args.task.upper() == "IMAGENET":
    loop_class_list = [2,100, 250, 500,750, 1000]
    loop_percent_list = [0,25,50,75,100]
else:
    loop_class_list = [2,3,5,7,10]
    loop_percent_list = [0,25,50,75,100]

# run experiments
for i in range(int(args.loops)):
    for num_classes in loop_class_list:
        for percent in loop_percent_list:
            print("----------------------------------------------")
            print("number of classes:", num_classes)
            print("shared percentage:", percent)

            experiment(num_classes,percent,task=args.task, n_epochs=int(args.epochs),gpu_num=int(args.gpu),batch_size=int(args.batch_size), masked=args.masked, savemodel=args.savemodels, adv_steps=int(args.adv_steps), download_data=args.download_data, adv_training=args.advtrain)

            print("----------------------------------------------")
