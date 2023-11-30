import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim
        # self.device=device
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        # self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device)
        # c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device) # cell state for LSTM
        # out,(hn, cn) = self.rnn(x, (h0, c0))
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device)
        out,hn = self.rnn(x, h0)        
        res = self.sig(self.fc(out))
        return res
