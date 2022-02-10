import torch
from torch import nn
from torch.nn import functional as F
import math

def gaussian_positive(m):
    # print(type(m))
    if type(m) == nn.LSTM:
        nn.init.uniform_(m.weight_ih_l0, 0, math.sqrt(1 / m.weight_ih_l0.shape[1]))
        nn.init.uniform_(m.weight_hh_l0, 0, math.sqrt(1 / m.weight_hh_l0.shape[1]))
        # m.weight_ih_l0.data.uniform_(0, math.sqrt(1/m.weight_ih_l0.shape[1]))
        print('init weights')

# https://pytorch.org/docs/stable/nn.init.html
def xavier_weights_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

def He_weights_init(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class modulation_netowrk(nn.Module):
    def __init__(self, input_dim, feature_dim, af=' '):
        super(modulation_netowrk, self).__init__()
        self.af = af
        # self.fc = nn.Linear(input_dim, feature_dim*feature_dim)
        self.fc = nn.Linear(input_dim, feature_dim)

    def forward(self, x):
        x = self.fc(x)
        if self.af == 'gelu':
            x = F.gelu(x)
        elif self.af == 'relu':
            x = F.relu(x)
        return x

class feature_network(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim1,
                 # hidden_dim2
                 af='relu'):
        super(feature_network, self).__init__()
        self.af = af
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        # self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.bn1 = nn.BatchNorm1d(hidden_dim1)
        # self.bn2 = nn.BatchNorm1d(hidden_dim2)
        # self.dropout = nn.Dropout()

    def forward(self, x):
        # x = self.dropout(x)
        x = self.fc1(x)
        if self.af == 'gelu':
            x = F.gelu(x)
        elif self.af == 'relu':
            x = F.relu(x)
        # x = self.bn1(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.bn2(x)
        return x

class task_network(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(task_network, self).__init__()
        self.fc1 = nn.Linear(feature_dim, class_num)
        # self.fc2 = nn.Linear(class_num, class_num)
        # self.model_feature = model_feature
        # self.dropout = nn.Dropout()

    def forward(self, x):
        # x = self.model_feature(x)
        x = self.fc1(x)
        # x_ce = self.fc2(x)
        return x
            # , x_ce

class dot_product(nn.Module):
    def __init__(self):
        super(dot_product, self).__init__()
    def forward(self, feature, modulation_feature):
        return torch.mul(feature, modulation_feature)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V,  attn_mask=None):
        # if 0 in torch.sqrt(K[:,-1]):
        #     scores = torch.matmul(Q, K.transpose(-1, -2))
        # else:
        #     scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(K[:,-1]) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # if attn_mask is not None:
        #     scores.masked_fill_(attn_mask, -1e9)
        scores = torch.matmul(Q, K.transpose(-1, -2))
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


