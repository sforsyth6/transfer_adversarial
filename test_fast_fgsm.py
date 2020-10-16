from free_adv_training.train_fgsm import *
from torchvision import transforms, models
from torchvision.datasets import CIFAR100
import torch
from advertorch.attacks import LinfPGDAttack

def accuracy(predictions, true_y):
    correct = 0

    for i in range(len(predictions)):
        if int(torch.argmax(predictions[i])) == int(true_y[i]):
            correct += 1

    accuracy = correct / len(true_y)
    return accuracy

model = models.resnet18()
model.fc = nn.Linear(512, 100)

mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

transform_test = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize(mean, std)])

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

train_data = CIFAR100("data/",transform=transform_train,download=True)
test_data = CIFAR100("data/", train=False, transform=transform_test,download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=2)

model = fast_free_adv_training(model, train_loader, epochs=1, mean=mean, std=std, device='cuda:1')

# new_model.eval()
model.eval()

preds = []
test_y = []
#
# adversary = LinfPGDAttack(
#     new_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=.3,
#     nb_iter=10, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
#     targeted=False)

for data in test_loader:
    preds.append(model(data[0].to('cuda:1')))
    test_y.append(data[1])

preds = torch.cat(preds,axis=0)
test_y = torch.cat(test_y,axis=0)

print(preds.shape)

print(accuracy(preds,test_y))
