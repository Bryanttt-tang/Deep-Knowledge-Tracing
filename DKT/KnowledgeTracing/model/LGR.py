import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

class LGR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LGR, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        res = self.sigmoid(out)
        return res

