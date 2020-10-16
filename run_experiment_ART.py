from torchvision.datasets import CIFAR100
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

from art.attacks import FastGradientMethod
from art.attacks import ProjectedGradientDescent
from art.classifiers import PyTorchClassifier
from art.utils import load_dataset

from cifar100 import CIFAR100

from metrics import accuracy
from metrics import accuracy_n


# Get command line arguments
parser = argparse.ArgumentParser(description='Run Experiment')

parser.add_argument('--gpu', type=int, default=0, help='Which GPU should be used')
parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=64, help = "Number of images per batch")
parser.add_argument('--lr-steps', type=int, default=2, help = 'Number of learning rate steps')

parser.add_argument('--num-shared-classes', type =int, default=2, help = "Number of classes shared between the two models")
parser.add_argument('--percent-shared-data', type=int, default=100, help = "Amount of training data to be shared between shared classes")

parser.add_argument('--attack', choices=["fgsm",'pgd'], default="fgsm", help = "Adversarial attack type")
parser.add_argument('--eps', default=0.2,type=float ,help = "Eps for attack")
parser.add_argument('--adv-steps', default=100, type=int, help= "Step for attack (pgd only)")


args = parser.parse_args()

n_epochs = args.epochs
lr_steps = args.lr_steps
batch_size = args.batch_size


num_shared_classes = args.num_shared_classes
percent_shared_data = args.percent_shared_data

attack_choice = args.attack
eps = args.eps
adv_steps = args.adv_steps

# Read Dataset

dataset = CIFAR100()

(x_train, y_train), (x_test, y_test), min_, max_ = dataset.load_dataset("./data/")
im_shape = x_train[0].shape


# Split Classes

# class made to get classes from one-hot encoding
def getClass(givenList):
    for i in range(len(givenList)):
        if givenList[i] == 1:
            return i


# Find the classes that will exist in each model
all_classes = set([getClass(item) for item in y_train])
shared_classes = random.sample(all_classes, num_shared_classes) # randomly find shared classes
split_classes = [c for c in all_classes if c not in shared_classes] # get classes not shared


if len(split_classes) % 2 == 1: # if we have an odd #, randomly remove one so that number of classes will be the same for each model
    split_classes.pop(random.randint(0, len(split_classes) - 1))

model1_split = random.sample(split_classes, len(split_classes) // 2)
model2_split = [c for c in split_classes if c not in model1_split]


model1_classes = model1_split
model2_classes = model2_split

model1_classes.sort()
model2_classes.sort()


print("shared classes: {}".format(shared_classes))
print("model1 classes: {}".format(model1_classes))
print("model2 classes: {}".format(model2_classes))


# Create New Datasets


model1_x_train = []
model1_y_train = []

model2_x_train = []
model2_y_train = []

shared_x_train = []
shared_y_train = []

# train data splits
for index in range(len(x_train)):

    current_class = getClass(y_train[index])

    # model 1
    if current_class in model1_classes:
        model1_x_train.append(x_train[index])
        model1_y_train.append(y_train[index])

    # model 2
    if current_class in model2_classes:
        model2_x_train.append(x_train[index])
        model2_y_train.append(y_train[index])


# Carry out datasplitting for shared classes and add to datasets

for shared_class in shared_classes:

    all_examples_x_train = []
    all_examples_y_train = []

    # get all examples of class
    for index in range(len(x_train)):
        current_class = getClass(y_train[index])

        if current_class == shared_class:
            all_examples_x_train.append(x_train[index])
            all_examples_y_train.append(y_train[index])


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
    model1_x_train = model1_x_train + model1_examples_x_train + shared_examples_x_train
    model1_y_train = model1_y_train + model1_examples_y_train + shared_examples_y_train

    model2_x_train = model2_x_train + model2_examples_x_train + shared_examples_x_train
    model2_y_train = model2_y_train + model2_examples_y_train + shared_examples_y_train


model1_x_test = []
model1_y_test = []

model2_x_test = []
model2_y_test = []

shared_x_test = []
shared_y_test = []


# test data splits
for index in range(len(x_test)):

    current_class = getClass(y_test[index])

    # model 1
    if current_class in model1_classes:
        model1_x_test.append(x_test[index])
        model1_y_test.append(y_test[index])

    # model 2
    if current_class in model2_classes:
        model2_x_test.append(x_test[index])
        model2_y_test.append(y_test[index])

    # shared classes for eval
    if current_class in shared_classes:
        shared_x_test.append(x_test[index])
        shared_y_test.append(y_test[index])


# DEBUG:
model1_classes_debug = set([getClass(item) for item in model1_y_train])
model2_classes_debug = set([getClass(item) for item in model2_y_train])
shared_classes_debug = set([getClass(item) for item in model2_y_train])


print("model{} classes: {}".format(1,model1_classes_debug))
print("model{} classes: {}".format(2,model2_classes_debug))



# convert to numpy array
model1_x_train = np.array([np.array(item) for item in model1_x_train])
model1_y_train = np.array([np.array(item) for item in model1_y_train])

model1_x_test = np.array([np.array(item) for item in model1_x_test])
model1_y_test = np.array([np.array(item) for item in model1_y_test])

model2_x_train = np.array([np.array(item) for item in model2_x_train])
model2_y_train = np.array([np.array(item) for item in model2_y_train])

model2_x_test = np.array([np.array(item) for item in model2_x_test])
model2_y_test = np.array([np.array(item) for item in model2_y_test])

shared_x_test = np.array([np.array(item) for item in shared_x_test])
shared_y_test = np.array([np.array(item) for item in shared_y_test])


print("data processed...")


# Model Training

# Get model (using ResNet50 for now)
model1 = models.resnet50()
model2 = models.resnet50()

model1.fc = nn.Linear(2048, 100)
model2.fc = nn.Linear(2048, 100)

criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters())

criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model2.parameters())


classifier1 = PyTorchClassifier(model=model1, clip_values=(min_, max_), loss=criterion1,
                               optimizer=optimizer1, input_shape=(3,32,32), nb_classes=100)

classifier2 = PyTorchClassifier(model=model2, clip_values=(min_, max_), loss=criterion2,
                               optimizer=optimizer2, input_shape=(3,32,32), nb_classes=100)


print("training...")
classifier1.fit(model1_x_train, model1_y_train, batch_size=batch_size, nb_epochs=n_epochs)
classifier2.fit(model2_x_train, model2_y_train, batch_size=batch_size, nb_epochs=n_epochs)

# evaluation
model1.eval()
model2.eval()

predictions = classifier1.predict(shared_x_test)
acc = accuracy(predictions,shared_y_test)
print('Accuracy of model1 on shared test examples: {}%'.format(acc * 100))

top_five_acc = accuracy_n(predictions,shared_y_test,5)
print('Top 5 accuracy of model1 on shared test examples: {}%'.format(top_five_acc * 100))


predictions = classifier2.predict(shared_x_test)
acc = accuracy(predictions,shared_y_test)
print('Accuracy of model2 on shared test examples: {}%'.format(acc * 100))

top_five_acc = accuracy_n(predictions,shared_y_test,5)
print('Top 5 accuracy of model2 on shared test examples: {}%'.format(top_five_acc * 100))

# Define attack based on model1

if attack_choice == "fgsm":
    attack = FastGradientMethod(classifier=classifier1, eps=eps)
else:
    attack = ProjectedGradientDescent(classifier=classifier1, eps=eps, max_iter=adv_steps)

print()

print("generating adversarial examples...")

# generate adv examples for model1 based on shared data
x_test_adv = attack.generate(x=shared_x_test)

# test adv examples generated from model1 on model1
predictions = classifier1.predict(x_test_adv)
acc = accuracy(predictions,shared_y_test)
print('Accuracy of model1 on adversarial test examples: {}%'.format(acc * 100))

top_five_acc = accuracy_n(predictions,shared_y_test,5)
print('Top 5 accuracy of model1 on adversarial test examples: {}%'.format(top_five_acc * 100))

# test adv examples generated from model1 on model2
predictions = classifier2.predict(x_test_adv)
acc = accuracy(predictions,shared_y_test)
print('Accuracy of model2 on adversarial test examples: {}%'.format(acc * 100))

top_five_acc = accuracy_n(predictions,shared_y_test,5)
print('Top 5 accuracy of model2 on adversarial test examples: {}%'.format(top_five_acc * 100))
