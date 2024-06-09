import torch
import pickle
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from utils.model import GCN
from tqdm import tqdm
from utils.feature import get_feature, write_aig, extract_feature_target, score_aig_baseline
import concurrent.futures
import wandb
import os
from time import time
from multiprocessing import Pool
import gc

import h5py

start_time = time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graphs = []
test_graphs = []

data_dir = 'data/task2/'
for file in os.listdir(data_dir)[:10]:
    if file.endswith('.pkl'):
        if file.startswith('multiplier'):
            continue
        # key = file.split('.')[0]
        # baseline = score_aig_baseline('./InitialAIG/train/' + key + '.aig')
        with open(data_dir + file, 'rb') as f:
            key = file.split('.')[0]
            train_feature, train_label, test_feature, test_label = pickle.load(f)
            print(f'{key} train: {len(train_feature)}, test: {len(test_feature)}, time: {time() - start_time}')
        for i in range(min(len(train_feature), 2500)):
            for j in range(len(train_feature[i])):
                # train_label[i][j] = (1 - train_label[i][j]) * baseline
                
                data = train_feature[i][j]
                node_type, num_inverted_predecessors, edge_index, nodes = data.values()

                # Convert node features to PyTorch tensor
                node_type_tensor = node_type.view(-1, 1).to(torch.float)
                num_inverted_predecessors_tensor = num_inverted_predecessors.view(-1, 1).to(torch.float)
                
                # Stack the node features along the second dimension
                x = torch.cat([node_type_tensor, num_inverted_predecessors_tensor], dim=1)
                # x = x.flatten()

                # Convert edge index to PyTorch tensor
                # edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
                edge_index_tensor = edge_index.to(torch.long)

                # Create Data object and append to the list of graphs
                graph_data = Data(x=x, edge_index=edge_index_tensor, y=torch.tensor([train_label[i][j]]))
                graphs.append(graph_data)
        for i in range(len(test_feature)):
            for j in range(len(test_feature[i])):
                # test_label[i][j] = (1 - test_label[i][j]) * baseline
                
                data = test_feature[i][j]
                node_type, num_inverted_predecessors, edge_index, nodes = data.values()

                # Convert node features to PyTorch tensor
                node_type_tensor = node_type.view(-1, 1).to(torch.float)
                num_inverted_predecessors_tensor = num_inverted_predecessors.view(-1, 1).to(torch.float)
                
                # Stack the node features along the second dimension
                x = torch.cat([node_type_tensor, num_inverted_predecessors_tensor], dim=1)
                # x = x.flatten()

                # Convert edge index to PyTorch tensor
                # edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
                edge_index_tensor = edge_index.to(torch.long)

                # Create Data object and append to the list of graphs
                graph_data = Data(x=x, edge_index=edge_index_tensor, y=torch.tensor([test_label[i][j]]))
                test_graphs.append(graph_data)
        
        print(f'{key}, time: {time() - start_time}')
        # break
        
    print(len(graphs))
        
del train_feature, train_label, test_feature, test_label

# Create DataLoader with batching and multiple workers
loader = DataLoader(graphs, batch_size=128, shuffle=True, num_workers=4)
print(len(loader))
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False, num_workers=4)

del graphs, test_graphs
gc.collect()

# Now `loader` can be used for training
# del train_features, train_labels, test_features, test_labels

model = GCN(num_node_features=2, hidden_channels=16, num_output_features=1).to(device)  # 修改为实际的特征数和输出特征数
optimizer = torch.optim.Adam(model.parameters(), lr=1e2, weight_decay=5e-8)

def train():
    model.train()
    total_loss = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).view(-1)
        target = data.y
        loss = F.mse_loss(out, target)  # 使用均方误差损失
        loss.backward()
        # tqdm.write(f'loss: {loss.item()}')
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(loader):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data).view(-1)
        target = data.y
        loss = F.mse_loss(out, target)  # 使用均方误差损失
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

wandb.init(project="GNN-IC-Design-Task2")

num_epochs = 200
best_val_loss = float('inf')
best_model_path = 'best_model_task2_1e-1.pth'

for epoch in range(num_epochs):
    train_loss = train()
    val_loss = test(test_loader)  # 这里只使用训练数据进行评估

    # Log metrics to wandb
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss
    })

    # 保存验证损失最低的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'New best model saved at epoch {epoch:03d} with validation loss: {val_loss:.4f}')

    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

# Finish the wandb run
wandb.finish()