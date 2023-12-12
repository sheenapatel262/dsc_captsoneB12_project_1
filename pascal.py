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


from torch_geometric.datasets import LRGBDataset

# Create the Pascal VOC Superpixels dataset
pasc_dataset = LRGBDataset(root='.', name='PascalVOC-SP')

# Access a single data point from the dataset
pasc_data = pasc_dataset[0]

# Print or inspect the data
print(pasc_data)
pasc_dic = {}

print(pasc_dataset)
print("number of graphs:\t\t",len(pasc_dataset))
print("number of classes:\t\t",pasc_dataset.num_classes)
print("number of node features:\t",pasc_dataset.num_node_features)
print("number of edge features:\t", pasc_dataset.num_edge_features)


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

train_dataset = pasc_dataset[:int(len(pasc_dataset) * 0.8)]
test_dataset = pasc_dataset[int(len(pasc_dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
model = GCN(num_features=pasc_dataset.num_node_features, num_classes=pasc_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)  # Using all nodes for training
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_nodes

    train_accuracy = correct / total
    #print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Train Accuracy: {train_accuracy:.4f}')

# Testing loop
model.eval()
correct = 0
total = 0
for data in test_loader:
    out = model(data)
    pred = out.argmax(dim=1)
    correct += pred.eq(data.y).sum().item()
    total += data.num_nodes

test_acc = correct / total
print('PASCAL GCN:')
print("Test Accuracy: {:.4f}".format(test_acc))
pasc_dic['GCN'] = f'{test_acc * 100:.2f}%'


# In[4]:


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

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


model = GIN(in_channels=pasc_dataset.num_node_features, hidden_channels = 16, out_channels=pasc_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        optimizer.zero_grad()
        # Pass the correct arguments to the model
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_nodes

    train_accuracy = correct / total
    #print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Train Accuracy: {train_accuracy:.4f}')


model.eval()
correct = 0
total = 0
for data in test_loader:
    # Pass the correct arguments to the model
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct += pred.eq(data.y).sum().item()
    total += data.num_nodes

test_acc = correct / total
print('PASCAL GIN:')
print("Test Accuracy: {:.4f}".format(test_acc))
pasc_dic['GIN'] = f'{test_acc * 100:.2f}%'


# In[ ]:


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.out = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

    
model = GAT(num_node_features=pasc_dataset.num_node_features, num_classes=pasc_dataset.num_classes, hidden_channels = 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        optimizer.zero_grad()
        # Pass the correct arguments to the model
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_nodes

    train_accuracy = correct / total
    #print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Train Accuracy: {train_accuracy:.4f}')


model.eval()
correct = 0
total = 0
for data in test_loader:
    # Pass the correct arguments to the model
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct += pred.eq(data.y).sum().item()
    total += data.num_nodes

test_acc = correct / total
print('PASCAL GAT:')
print("Test Accuracy: {:.4f}".format(test_acc))
pasc_dic['GAT'] = f'{test_acc * 100:.2f}%'

