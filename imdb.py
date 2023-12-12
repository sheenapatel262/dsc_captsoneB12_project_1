#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[6]:


imdb_dataset = TUDataset(root='.', name='IMDB-BINARY')
print(imdb_dataset)


# In[9]:


print("First Graph in IMBD Dataset with 0 features:")
print("Number of nodes:", imdb_dataset[0].num_nodes)
print("Number of edges:", imdb_dataset[0].num_edges)
num_node_features = imdb_dataset[0].num_features
print("Number of Node Features:", num_node_features)
print("Edge indices shape:", imdb_dataset[0].edge_index.shape)
print("Graph label:", imdb_dataset[0].y.item())


# In[10]:


modified_imdb_dataset = []
for data in imdb_dataset:
    num_nodes = data.num_nodes
    constant_feature = torch.ones((num_nodes, 1))
    edge_index = data.edge_index
    deg = degree(edge_index[0], num_nodes).view(-1, 1).float()
    data.x = torch.cat([constant_feature, deg], dim=1)
    modified_imdb_dataset.append(data)
    
print("First Graph in Modified IMBD Dataset:")
print("Number of nodes:", modified_imdb_dataset[0].num_nodes)
print("Number of edges:", modified_imdb_dataset[0].num_edges)
print("Node features shape:", modified_imdb_dataset[0].x.shape)
print("Edge indices shape:", modified_imdb_dataset[0].edge_index.shape)
print("Graph label:", modified_imdb_dataset[0].y.item())

# Number of node features in the first graph
num_node_features = modified_imdb_dataset[0].num_features
print("Number of Node Features:", num_node_features)

# Number of classes in the dataset
num_classes = len(set([data.y.item() for data in modified_imdb_dataset]))
print("Number of Classes:", num_classes)


# In[11]:


imdb_dic = {}


# In[12]:


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.out(x)       
        return x
    
loader = DataLoader(modified_imdb_dataset, batch_size=32, shuffle=True)
model = GCN(num_node_features=num_node_features, num_classes=num_classes, hidden_channels=64)
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
print('IMDB GCN:')
print(f'Test Accuracy: {test_acc * 100:.2f}%')
imdb_dic['GCN'] = f'{test_acc * 100:.2f}%'


# In[13]:


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.out = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.out(x)
        x  = F.log_softmax(x, dim=1)
        return x

loader = DataLoader(modified_imdb_dataset, batch_size=32, shuffle=True)
model = GAT(num_node_features=num_node_features, num_classes=num_classes, hidden_channels=8)
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
print('IMDB GAT:')
print(f'Test Accuracy: {test_acc * 100:.2f}%')
imdb_dic['GAT'] = f'{test_acc * 100:.2f}%'


# In[14]:


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(GIN, self).__init__()
        mlp1 = Sequential(
            Linear(2, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels), 
            BatchNorm1d(hidden_channels)
        )
        self.conv1 = GINConv(mlp1)
        mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels), 
            BatchNorm1d(hidden_channels)
        )
        self.conv2 = GINConv(mlp2)
        self.lin = Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x
    
loader = DataLoader(modified_imdb_dataset, batch_size=32, shuffle=True)
model = GIN(hidden_channels=64, num_classes=num_classes)
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
print('IMDB GIN:')
print(f'Test Accuracy: {test_acc * 100:.2f}%')
imdb_dic['GIN'] = f'{test_acc * 100:.2f}%'


# In[15]:


from torch_geometric.nn import global_mean_pool 
from sklearn.model_selection import train_test_split
from torch_geometric.nn import global_mean_pool 
from torch_geometric.transforms import OneHotDegree, NormalizeFeatures

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

        # Add a final linear layer for classification
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

transform = OneHotDegree(max_degree=135)
dataset = TUDataset(root='/tmp/IMDB', name='IMDB-BINARY', transform=transform)
data = dataset[0]

model = GPSConvNet(dataset.num_node_features, hidden_channels, dataset.num_classes, heads=1, dropout=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

from torch_geometric.loader import DataLoader

train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

for epoch in range(50):
    loss = train()

loader = DataLoader(dataset, batch_size=32, shuffle=True)
test_acc = test(loader)
print('IMDB GraphGPS:')
print(f'Test Accuracy: {test_acc:f}')
imdb_dic['GPS'] = f'{test_acc * 100:.2f}%'

