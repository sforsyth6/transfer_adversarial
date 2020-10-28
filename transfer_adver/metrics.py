import numpy as np


def accuracy(predictions, true_y):
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(true_y, axis=1)) / len(true_y)
    return accuracy

def accuracy_n(predictions, true_y,n):
    # get the top 5 most likley classes for each instance based on model
    top_n = [np.argpartition(instance, -n)[-n:] for instance in predictions]
    # get the true label for each instance
    top_target = np.argmax(true_y, axis=1)
    total = 0
    # for every single one of the top five
    for i, j in enumerate(top_n):
        # if the true label for the instance is in the top five prediction
        if top_target[i] in j:
            total += 1
    # divide by total number of instances
    top_n_acc = float(total) / len(top_target)

    return top_n_acc
