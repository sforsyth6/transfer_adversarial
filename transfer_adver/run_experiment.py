from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageNet
from torchvision.datasets import FashionMNIST
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
import time


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

def experiment(num_shared_classes, percent_shared_data, n_epochs=200,batch_size=128, eps=.3, adv_steps=100, learning_rate=.0004, gpu_num=1,adv_training=False,task="CIFAR100"):
    print("epochs,batch_size,eps,adv_steps,learning_rate,task")
    print(n_epochs,batch_size,eps,adv_steps,learning_rate,task)

    cuda = torch.cuda.is_available()

    transform_test = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
            ])

    if task.upper() == "CIFAR100":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_data = CIFAR100("data/",transform=transform_train, download=False)
        test_data = CIFAR100("data/", train=False, transform=transform_test, download=False)
    elif task.upper() == "IMAGENET":
        train_data = ImageNet('data/imagenet', split='train', download=False)
        test_data = ImageNet('data/imagenet', split='val', download=False)
    elif task.upper() == "FASHIONMNIST":
        transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                        transforms.ToTensor()
                             ])

        train_data = FashionMNIST('data/fashionmnist',transform=transform, train=True, download=False)
        test_data = FashionMNIST('data/fashionmnist', transform=transform, train=False, download=False)
    else:
        train_data = CIFAR10("data/",transform=transform_train,download=False)
        test_data = CIFAR10("data/", train=False, transform=transform_test,download=False)

        # model1 = ResNet(ResidualBlock, [2, 2, 2],num_classes=10)
        # model2 = ResNet(ResidualBlock, [2, 2, 2],num_classes=10)


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

    model1_x_train = []
    model1_y_train = []

    model2_x_train = []
    model2_y_train = []

    shared_x_train = []
    shared_y_train = []

    # train data splits
    for index in range(len(train_data)):

        current_class = train_data[index][1]

        # model 1
        if current_class in model1_classes:
            model1_x_train.append(train_data[index][0])
            model1_y_train.append(train_data[index][1])

        # model 2
        if current_class in model2_classes:
            model2_x_train.append(train_data[index][0])
            model2_y_train.append(train_data[index][1])


    # split by percentage for classes per model1

    if percent_shared_data < 100:

        new_model1_x_train = []
        new_model1_y_train = []

        for curr_class in model1_classes:
            temp_data_x = []
            temp_data_y = []

            # get all examples of class
            for i in range(len(model1_x_train)):
                if(model1_y_train[i] == curr_class):
                    temp_data_x.append(model1_x_train[i])
                    temp_data_y.append(model1_y_train[i])

            # split data by half the size
            total_size = len(temp_data_x)
            shared_size = int(total_size * .5)

            shared_indices = random.sample(list(range(len(temp_data_x))),shared_size)

            new_model1_x_train += [temp_data_x[i] for i in shared_indices]
            new_model1_y_train += [temp_data_y[i] for i in shared_indices]


        # split for model2

        new_model2_x_train = []
        new_model2_y_train = []

        for curr_class in model2_classes:
            temp_data_x = []
            temp_data_y = []

            # get all examples of class
            for i in range(len(model2_x_train)):
                if(model2_y_train[i] == curr_class):
                    temp_data_x.append(model2_x_train[i])
                    temp_data_y.append(model2_y_train[i])

            # split data by half the size
            total_size = len(temp_data_x)
            shared_size = int(total_size * .5)

            shared_indices = random.sample(list(range(len(temp_data_x))),shared_size)

            new_model2_x_train += [temp_data_x[i] for i in shared_indices]
            new_model2_y_train += [temp_data_y[i] for i in shared_indices]


        # rewrite dataset
        model1_x_train = new_model1_x_train
        model1_y_train = new_model1_y_train

        model2_x_train = new_model2_x_train
        model2_y_train = new_model2_y_train

    # Carry out datasplitting for shared classes and add to datasets

    for shared_class in shared_classes:

        all_examples_x_train = []
        all_examples_y_train = []

        # get all examples of class
        for index in range(len(train_data)):
            current_class = train_data[index][1]

            if current_class == shared_class:
                all_examples_x_train.append(train_data[index][0])
                all_examples_y_train.append(train_data[index][1])


        # find max number of samples per model (set to be amount of examples if data is completely disjoint)
        max_examples = len(all_examples_x_train) // 2

        # get shared examples
        shared_examples_x_train = []
        shared_examples_y_train = []

        num_shared_examples = max_examples * percent_shared_data // 100
        for _ in range(num_shared_examples):
            random_int = random.randint(0, len(all_examples_x_train) - 1)

            shared_examples_x_train.append(all_examples_x_train.pop(random_int))
            shared_examples_y_train.append(all_examples_y_train.pop(random_int))


        # get disjoint examples
        disjoint_examples = max_examples - len(shared_examples_x_train)

        model1_examples_x_train = []
        model1_examples_y_train = []

        model2_examples_x_train = []
        model2_examples_y_train = []

        for _ in range(disjoint_examples):
            model1_rand_int = random.randint(0, len(all_examples_x_train) - 1)

            model1_examples_x_train.append(all_examples_x_train.pop(model1_rand_int))
            model1_examples_y_train.append(all_examples_y_train.pop(model1_rand_int))

            model2_rand_int = random.randint(0, len(all_examples_x_train) - 1)
            model2_examples_x_train.append(all_examples_x_train.pop(model2_rand_int))
            model2_examples_y_train.append(all_examples_y_train.pop(model2_rand_int))


        # add to the datasets for the model
        model1_x_train = shared_examples_x_train + model1_x_train + model1_examples_x_train
        model1_y_train = shared_examples_y_train + model1_y_train + model1_examples_y_train

        model2_x_train = shared_examples_x_train + model2_x_train + model2_examples_x_train
        model2_y_train = shared_examples_y_train + model2_y_train + model2_examples_y_train

    #print(model1_y_train)

    # assign mapping for new classes
    model1_class_mapping = {}
    model2_class_mapping = {}

    model1_classes_inc = 0
    # go through model1 and assign unique classes to incrimental int starting at 0
    for index in range(len(model1_y_train)):
        # if it doesn't exist assign
        if model1_y_train[index] not in model1_class_mapping.keys():
            model1_class_mapping[model1_y_train[index]] = model1_classes_inc
            model1_classes_inc += 1
        # append assigned token
        model1_y_train[index] = model1_class_mapping[model1_y_train[index]]


    model2_classes_inc = 0
    # go through model2 and assign unique classes to incrimental int starting at 0
    for index in range(len(model2_y_train)):
        # if it doesn't exist in model2 OR in model1, assign it
        if model2_y_train[index] not in model2_class_mapping.keys() and model2_y_train[index] not in model1_class_mapping.keys():
            model2_class_mapping[model2_y_train[index]] = model2_classes_inc
            model2_y_train[index] = model2_classes_inc
            model2_classes_inc += 1
        elif model2_y_train[index] in model1_class_mapping.keys():
            model2_y_train[index] = model1_class_mapping[model2_y_train[index]]
        else:
            model2_y_train[index] = model2_class_mapping[model2_y_train[index]]

    model1_x_test = []
    model1_y_test = []

    model2_x_test = []
    model2_y_test = []

    shared_x_test = []
    shared_y_test = []


    # test data splits
    for index in range(len(test_data)):

        current_class = test_data[index][1]

        # model 1
        if current_class in model1_classes:
            model1_x_test.append(test_data[index][0])
            model1_y_test.append(test_data[index][1])

        # model 2
        if current_class in model2_classes:
            model2_x_test.append(test_data[index][0])
            model2_y_test.append(test_data[index][1])

        # shared classes for eval
        if current_class in shared_classes:
            shared_x_test.append(test_data[index][0])
            shared_y_test.append(test_data[index][1])

    model1_x_test += shared_x_test
    model1_y_test += shared_y_test

    model2_x_test += shared_x_test
    model2_y_test += shared_y_test


    for index in range(len(model1_y_test)):
        model1_y_test[index] = model1_class_mapping[model1_y_test[index]]


    for index in range(len(model2_y_test)):
        if model2_y_test[index] in model1_class_mapping.keys():
            model2_y_test[index] = model1_class_mapping[model2_y_test[index]]
        else:
            model2_y_test[index] = model2_class_mapping[model2_y_test[index]]


    model1_classes_len= len(set([item for item in model1_y_train]))
    model2_classes_len = len(set([item for item in model2_y_train]))


    if task.upper() == "CIFAR100":

        model1 = models.wide_resnet50_2()
        model2 = models.wide_resnet50_2()
        #
        model1.fc = nn.Linear(2048, model1_classes_len)
        model2.fc = nn.Linear(2048, model2_classes_len)

    elif task.upper() == "IMAGENET":
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


    cuda = torch.cuda.is_available()
    if gpu_num in range(torch.cuda.device_count()):
        device = torch.device('cuda:'+str(gpu_num) if cuda else 'cpu')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # Model Training

    model1 = model1.to(device)
    model2 = model2.to(device)

    criterion1 = nn.CrossEntropyLoss()
    optimizer1 = optim.AdamW(model1.parameters(), lr=learning_rate)
    scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1,milestones=[60, 120, 160], gamma=.2) #learning rate decay


    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = optim.AdamW(model2.parameters(), lr=learning_rate)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2,milestones=[60, 120, 160], gamma=.2) #learning rate decay

    # zip together two lists
    train_set1 = list(zip(model1_x_train, model1_y_train))

    # create trainloader 1
    trainloader_1 = torch.utils.data.DataLoader(train_set1, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    # create trainloader 2

    # zip together two lists
    train_set2 = list(zip(model2_x_train, model2_y_train))

    # create trainloader 1
    trainloader_2 = torch.utils.data.DataLoader(train_set2, batch_size=batch_size,
                                              shuffle=True, num_workers=2)


    # TODO change this
    num_adv_batchs = 2 if adv_training else 0

    adv_batches = random.sample(range(len(trainloader_1)), num_adv_batchs)

    #print("adv_batches:", adv_batches)

    # train model 1
    for epoch in tqdm(range(n_epochs),desc="Epoch"):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader_1, 0):
            if cuda:
                data = tuple(d.cuda() for d in data)


            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer1.zero_grad()

            # forward + backward + optimize

            # train adversarial
    #         if i in adv_batches:
    #             print("adv training!")
    #             adversary = LinfPGDAttack(
    #                 model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
    #                 nb_iter=adv_steps, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    #                 targeted=False)
    #             inputs = adversary.perturb(inputs, labels)


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
            if cuda:
                data = tuple(d.cuda() for d in data)

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

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

    model1 = model1.to("cpu")
    model2 = model2.to("cpu")


    # convert shared classes to new labels
    for index in range(len(shared_y_test)):
        if shared_y_test[index] in model1_class_mapping.keys():
            shared_y_test[index] = model1_class_mapping[shared_y_test[index]]
        else:
            shared_y_test[index] = model2_class_mapping[shared_y_test[index]]


    shared_y_test = torch.Tensor(shared_y_test).long()


    # if cuda:
    #     shared_x_test = tuple(d.cuda() for d in shared_x_test)
    #     shared_y_test = torch.Tensor(shared_y_test).long().cuda()

    model1_x_test = torch.stack(model1_x_test)
    model2_x_test = torch.stack(model2_x_test)

    model1.eval()

    shared_x_test = torch.stack(shared_x_test)

    model1.eval()

    adversary = LinfPGDAttack(
        model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
        nb_iter=adv_steps, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

    adv_untargeted = adversary.perturb(shared_x_test, shared_y_test)

    timestr = time.strftime("%Y%m%d_%H%M%S")

    print("saving models at", timestr)

    model1_name = './models/{}_{}_{}_model1_{}.pickle'.format(task,num_shared_classes, percent_shared_data,timestr)
    model2_name = './models/{}_{}_{}_model2_{}.pickle'.format(task,num_shared_classes, percent_shared_data,timestr)
    adv_name = './models/{}_{}_{}_adv_{}.pickle'.format(task,num_shared_classes, percent_shared_data,timestr)


    torch.save(model1, model1_name)
    torch.save(model2, model2_name)
    torch.save(adversary, adv_name)

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


def pretty_experiment(num_classes,percent,task="CIFAR100",epochs=200,gpu_num=1,batch_size=128):
    print("----------------------------------------------")
    print("number of classes:", num_classes)
    print("shared percentage:", percent)

    experiment(num_shared_classes=num_classes, percent_shared_data=percent,task=task,n_epochs=epochs,gpu_num=gpu_num, batch_size=batch_size, adv_steps=1000)
    print("----------------------------------------------")


import argparse

parser = argparse.ArgumentParser(description='Experiment')

parser.add_argument('--task', default="CIFAR100")
parser.add_argument('--epochs', default=200)
parser.add_argument('--gpu', default=0)
parser.add_argument('--batch_size', default=128)

args = parser.parse_args()

# run experiments
for i in range(5):
    for num_classes in [2,50,100]:
        for percent in [0,50,100]:
            pretty_experiment(num_classes,percent,task=args.task,epochs=int(args.epochs),gpu_num=int(args.gpu),batch_size=int(args.batch_size))
