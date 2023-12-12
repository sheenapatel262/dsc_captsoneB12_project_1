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


# In[3]:


enz_dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)
print(enz_dataset)
enz_dic = {}


# In[4]:


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.bn1 = BatchNorm1d(64)
        self.conv2 = GCNConv(64, 32)
        self.bn2 = BatchNorm1d(32)
        self.lin = Linear(32, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x
    
loader = DataLoader(enz_dataset, batch_size=32, shuffle=True)
model = GCN(num_node_features=enz_dataset.num_node_features, num_classes=enz_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    model.train()
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
test_corr = 0
for data in loader: 
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1) 
    test_corr += int((pred == data.y).sum())
test_acc = test_corr / len(loader.dataset)
print(f'Enzymes GCN:')
print(f'Test Accuracy: {test_acc * 100:.2f}%')
enz_dic['GCN'] = f'{test_acc * 100:.2f}%'


# In[5]:


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x
    
loader = DataLoader(enz_dataset, batch_size=32, shuffle=True)
model = GAT(in_channels=enz_dataset.num_node_features, hidden_channels=32, out_channels=enz_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    model.train()
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
test_corr = 0
for data in loader: 
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1) 
    test_corr += int((pred == data.y).sum())
test_acc = test_corr / len(loader.dataset)
print(f'Enzymes GAT:')
print(f'Test Accuracy: {test_acc * 100:.2f}%')
enz_dic['GAT'] = f'{test_acc * 100:.2f}%'


# In[6]:


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.mlp1 = Sequential(
            Linear(in_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels), 
            BatchNorm1d(hidden_channels)
        )
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels)
        )
        self.conv2 = GINConv(self.mlp2)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x
    
loader = DataLoader(enz_dataset, batch_size=32, shuffle=True)
model = GIN(in_channels=enz_dataset.num_node_features, hidden_channels=64, out_channels=enz_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    model.train()
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
test_corr = 0
for data in loader: 
    out = model(data.x, data.edge_index, data.batch)  
    pred = out.argmax(dim=1) 
    test_corr += int((pred == data.y).sum())
test_acc = test_corr / len(loader.dataset)
print(f'Enzymes GIN:')
print(f'Test Accuracy: {test_acc * 100:.2f}%')
enz_dic['GIN'] = f'{test_acc * 100:.2f}%'


# In[7]:


class GPSConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(
            Linear(in_channels, 2 * h),
            nn.GELU(),
            Linear(2 * h, h),
            nn.GELU(),
        )
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)

        self.final_lin = Linear(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index, batch):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)

hidden_channels = [64, 64, 64]

dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)

model = GPSConvNet(dataset.num_node_features, hidden_channels, dataset.num_classes, heads=1, dropout=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

from torch_geometric.loader import DataLoader

train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum().item())
        total += data.num_graphs
    return correct / total
for epoch in range(5):
    loss = train()
    #print(f'Epoch {epoch}: Loss {loss:.4f}')
test_acc = test(test_loader)
print(f'Enzymes Graph GPS:')
print(f'Test Accuracy: {test_acc:.4f}')
enz_dic['GPS'] = f'{test_acc * 100:.2f}%'

