import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

class LGR(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LGR, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device)
    #     # out,hn = self.rnn(x, h0)
    #     hidden = self.tanh(self.hidden_layer(x))
        
    #     # Apply sigmoid activation for binary classification
    #     res = self.sigmoid(self.output_layer(hidden))
        
    #     return res

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device)
        # out,hn = self.rnn(x, h0)
        hidden = self.hidden_layer(x)
        
        # Apply sigmoid activation for binary classification
        res = self.sigmoid(self.output_layer(hidden))
        
        return res
