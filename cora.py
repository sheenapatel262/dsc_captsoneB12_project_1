#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, global_mean_pool, global_add_pool
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
import pandas as pd
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GATv2Conv
from torch_geometric.datasets import Planetoid
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GATv2Conv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Linear
import torch


# In[2]:


cora_dataset = Planetoid(root='.', name='Cora')
cora_data = cora_dataset[0]
cora_data


# In[3]:


print(cora_dataset)
print("number of graphs:\t\t",len(cora_dataset))
print("number of classes:\t\t",cora_dataset.num_classes)
print("number of node features:\t",cora_dataset.num_node_features)
print("number of edge features:\t", cora_dataset.num_edge_features)


# In[6]:


cora_dic = {}


# In[34]:


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(cora_dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, cora_dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
model = GCN(hidden_channels= 20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(cora_data.x, cora_data.edge_index)
    loss = criterion(out[cora_data.train_mask], cora_data.y[cora_data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    out = model(cora_data.x, cora_data.edge_index)
    pred = out.argmax(dim=1)
    test_corr = pred[cora_data.test_mask] == cora_data.y[cora_data.test_mask]
    test_acc = int(test_corr.sum()) / int(cora_data.test_mask.sum())
    train_corr = pred[cora_data.train_mask] == cora_data.y[cora_data.train_mask]
    train_acc = int(train_corr.sum()) / int(cora_data.train_mask.sum())
    #print(f'Test Accuracy: {test_acc * 100:.2f}%')
    cora_dic['GCN'] = f'{test_acc * 100:.2f}%'
print('GCN Cora:')
print(f'Train Accuracy: {train_acc * 100:.2f}%')
print(f'Test Accuracy: {test_acc * 100:.2f}%')


# In[38]:


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()  
        self.conv1 = GATv2Conv(cora_dataset.num_features, 8, heads = 8, dropout=0.6)
        self.conv2 = GATv2Conv(8 * 8, cora_dataset.num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p = 0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p = 0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim =1 )
        return x
    
model = GAT()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(cora_data.x, cora_data.edge_index)
    loss = criterion(out[cora_data.train_mask], cora_data.y[cora_data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    out = model(cora_data.x, cora_data.edge_index)
    pred = out.argmax(dim=1)
    train_corr = pred[cora_data.train_mask] == cora_data.y[cora_data.train_mask]
    train_acc = int(train_corr.sum()) / int(cora_data.train_mask.sum())
    test_corr = pred[cora_data.test_mask] == cora_data.y[cora_data.test_mask]
    test_acc = int(test_corr.sum()) / int(cora_data.test_mask.sum())
    #print(f'Test Accuracy: {test_acc * 100:.2f}%')
    cora_dic['GAT'] = f'{test_acc * 100:.2f}%'

print('GAT Cora:')
print(f'Train Accuracy: {train_acc * 100:.2f}%')
print(f'Test Accuracy: {test_acc * 100:.2f}%')


# In[36]:


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate = 0.6):
        super(GIN, self).__init__()
        mlp1 = Sequential(
            Linear(input_dim, hidden_dim), 
            #ReLU(), 
            #Linear(hidden_dim, hidden_dim), 
            BatchNorm1d(hidden_dim)
        )
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.bn1 = BatchNorm1d(hidden_dim)
        mlp2 = Sequential(
            Linear(hidden_dim, hidden_dim), 
            #ReLU(), 
            #Linear(hidden_dim, hidden_dim), 
            BatchNorm1d(hidden_dim)
        )
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.linear_prediction = Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear_prediction(x)
        return x
    
model = GIN(
    input_dim=cora_dataset.num_node_features, 
    hidden_dim = 100, 
    output_dim=cora_dataset.num_classes
)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05, weight_decay = .001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(cora_data.x, cora_data.edge_index)
    loss = criterion(out[cora_data.train_mask], cora_data.y[cora_data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    out = model(cora_data.x, cora_data.edge_index)
    pred = out.argmax(dim=1)
    train_corr = pred[cora_data.train_mask] == cora_data.y[cora_data.train_mask]
    train_acc = int(train_corr.sum()) / int(cora_data.train_mask.sum())
    test_corr = pred[cora_data.test_mask] == cora_data.y[cora_data.test_mask]
    test_acc = int(test_corr.sum()) / int(cora_data.test_mask.sum())
    #print(f'Test Accuracy: {test_acc * 100:.2f}%')
    cora_dic['GIN'] = f'{test_acc * 100:.2f}%'

print('GIN Cora:')
print(f'Train Accuracy: {train_acc * 100:.2f}%')
print(f'Test Accuracy: {test_acc * 100:.2f}%')


# In[35]:


class GPSConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        prev_channels = in_channels
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(
            Linear(in_channels, 2 * h),
            nn.GELU(),
            Linear(2 * h, h),
            nn.GELU(),
        )
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout = dropout, act=act)
            self.GPSConvs.append(gps_conv)
            prev_channels = h
        self.final_conv = GATv2Conv(prev_channels, out_channels, heads=1, dropout=dropout)
    
    def forward(self, x, edge_index):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.final_conv(x, edge_index)
        return F.log_softmax(x, dim=1)

hidden_channels = [64, 64, 64]
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = GPSConvNet(dataset.num_node_features, hidden_channels, dataset.num_classes, heads=1, dropout=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    logits = model(data.x, data.edge_index)
    accs = [torch.sum(logits[mask].argmax(dim=1) == data.y[mask]).item() / mask.sum().item() for mask in [data.train_mask, data.val_mask, data.test_mask]]
    return accs

for epoch in range(50):
    loss = train()
    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test()
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              #f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

print('GraphGPS Cora:')
print(f'Train Accuracy: {train_acc * 100:.2f}%')
print(f'Test Accuracy: {test_acc * 100:.2f}%')
cora_dic['GPS'] = f'{test_acc * 100:.2f}%'

