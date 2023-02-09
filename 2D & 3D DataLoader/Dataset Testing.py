import torch
import torchvision
import torchvision.datasets as datasets

# mnist data is list of tuple of (pil image, integer)
# mnist data is 28x28, and black and white (so 1x28x28)
data_dir = 'dataset'
mnist_trainset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
print(len(mnist_trainset))
print(type(mnist_trainset))
print(type(mnist_trainset[0]))

mnist_testset = datasets.MNIST(data_dir, train = False, download = True)
print(len(mnist_testset))