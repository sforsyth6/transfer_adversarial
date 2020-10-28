import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")

fashionmnist_num_classes = [2,2,2,50,50,50,100,100,100]
fashionmnist_percent = [0,50,100,0,50,100,0,50,100]
fashionmnist_success = [0.3412,0.2684,0.4258,0.38688,0.41056,0.41476,0.45736,0.48844,0.5119]

cifar10_num_classes = [2,2,2,50,50,50,100,100,100]
cifar10_percent = [0,50,100,0,50,100,0,50,100]
cifar10_success = [0.294,
    0.2037,
    0.2499,
    0.34944,
    0.28124,
    0.2452,
    0.31606,
    0.33126,
    0.32286]

cifar100_num_classes = [2,2,2,50,50,50,100,100,100]
cifar100_percent = [0,50,100,0,50,100,0,50,100]
cifar100_success = [0.229,0.226,0.332,0.2408,0.25268,0.24084,0.24028,0.25056,0.2538]


fashionmnist_success = [100**(x*10) for x in fashionmnist_success]

sns.scatterplot(x=cifar10_num_classes, y=cifar10_success, size=cifar10_percent)

plt.xlabel("Number Shared Classes")
# plt.ylabel("Percent of Shared Data")
plt.ylabel("Sucess of Transfer Attack")
plt.title("Success of Transfer Adversarial Attack")

# plt.savefig("cifar10_scatterplot.png")

plt.show()
