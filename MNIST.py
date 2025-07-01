import torch
import torchvision

class MNIST():
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def preprocess(self):
        # Example preprocessing step
        self.data = self.data / 255.0  # Normalize pixel values
        return self.data