import torch
import torchvision
from torchvision import transforms
from torch.utils import data

class MNIST():
    def __init__(self):
        self.trans = transforms.ToTensor()
        self.mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=self.trans ,download=True)
        self.mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=self.trans ,download=True)

    def get_data(self):
        return self.data

    def preprocess(self):
        # Example preprocessing step
        self.data = self.data / 255.0  # Normalize pixel values
        return self.data