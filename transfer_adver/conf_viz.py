cifar10_num_classes = [2,5,10]
cifar10_percent = [0,50,100]
cifar10_success = [[0.294,0.2037,0.2499],[0.34944,0.28124,0.2452],[0.31606,0.33126,0.32286]]

fashionmnist_num_classes = [2,5,10]
fashionmnist_percent = [0,50,100]
fashionmnist_success = [[0.3412,0.2684,0.4258],[0.38688,0.41056,0.41476],[0.45736,0.48844,0.5119]]

cifar100_num_classes = [25,50,75,100]
cifar100_percent = [0,25,50,75,100]
cifar100_success = [[0.41024,0.47136,0.4636,0.49368,0.53744],[0.39828,0.43952,0.45584,0.49172,0.48604],[0.382026667,0.428426667,0.459013333,0.470746667,0.4912],[0.3864,0.419,0.4359,0.47244,0.47684]]

cifar100_std = [[0.042544659,0.01802354,0.016052414,0.042186633,0.02432546], [0.010169169,0.020592523,0.012142817,0.02592088,0.026867601], [0.011695165,0.01583734,0.013166489,0.019579036,0.019688914], [0.009241212,0.008413085,0.004641121,0.005012285,0.009987142]]


masked_cifar100_success = [[0.3136,0.4208,0.396,0.3884,0.4152],[0.3206,0.366,0.3866,0.3844,0.407],[0.319466667,0.3552,0.3988,0.4132,0.412],[0.313,0.3691,0.3738,0.3944,0.4131]]


advhard_cifar100_success = [[0.3876, 0.3712, 0.4808, 0.4132, 0.48], [0.3044, 0.393, 0.3982, 0.4506, 0.456], [0.324666667, 0.3972, 0.412666667, 0.424933333, 0.438666667], [0.3168, 0.3702, 0.3977, 0.4172, 0.4402]]

cifar100_sucess_250 = [[0.3212,0.5128,0.4428,0.54,0.4904],[0.412,0.4198,0.5224,0.4974,0.5032],[0.383866667,0.4276,0.455333333,0.499333333,0.490266667],[0.3997,0.4184,0.4535,0.4804,0.4885]]


import seaborn as sns
import matplotlib.pyplot as plt

#
# sns.set(font_scale=3)
# plt.figure(figsize=(25,15))
# sns.heatmap(cifar100_success, cmap='OrRd',annot=True, xticklabels=cifar100_percent, yticklabels=cifar100_num_classes)
# plt.ylabel("Number Shared Classes")
# plt.xlabel("Percent of Shared Data")
# plt.title("Success of Transfer Adversarial Attack CIFAR100")
# plt.savefig("cifar100_cfnmtrx.png")
# plt.show()
#
# sns.heatmap(cifar10_success, cmap='OrRd',annot=True, xticklabels=cifar10_percent, yticklabels=cifar10_num_classes)
# plt.ylabel("Number Shared Classes")
# plt.xlabel("Percent of Shared Data")
# plt.title("Success of Transfer Adversarial Attack CIFAR10")
# plt.savefig("cifar10_cfnmtrx.png")
# plt.show()
#
# sns.heatmap(fashionmnist_success, cmap='OrRd',annot=True, xticklabels=fashionmnist_percent, yticklabels=fashionmnist_num_classes)
# plt.ylabel("Number Shared Classes")
# plt.xlabel("Percent of Shared Data")
# plt.title("Success of Transfer Adversarial Attack FashionMNIST")
# plt.savefig("fashionmnist_cfnmtrx.png")
# plt.show()


sns.heatmap(cifar100_std, cmap='Blues',annot=True, xticklabels=cifar100_percent, yticklabels=cifar100_num_classes)
plt.ylabel("Number Shared Classes")
plt.xlabel("Percent of Shared Data")
plt.title("Std. Dev. of Success of Transfer Adversarial Attack CIFAR100")
plt.savefig("cifar100_cfnmtrx_std.png")
plt.show()

#
# sns.heatmap(masked_cifar100_success, cmap='OrRd',annot=True, xticklabels=cifar100_percent, yticklabels=cifar100_num_classes)
# plt.ylabel("Number Shared Classes")
# plt.xlabel("Percent of Shared Data")
# plt.title("Success of Transfer Adversarial Masked Attack CIFAR100")
# plt.savefig("cifar100_cfnmtrx_masked.png")
# plt.show()
#
#
# sns.heatmap(advhard_cifar100_success, cmap='OrRd',annot=True, xticklabels=cifar100_percent, yticklabels=cifar100_num_classes)
# plt.ylabel("Number Shared Classes")
# plt.xlabel("Percent of Shared Data")
# plt.title("Success of Transfer Adversarial Attack CIFAR100 (Hardened)")
# plt.savefig("cifar100_cfnmtrx_advhard.png")
# plt.show()
