import torch
import pickle
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from utils.model import GCN
from tqdm import tqdm
from utils.feature import get_feature, write_aig, extract_feature_target
import concurrent.futures

with open('datas.pkl', 'rb') as f:
    datas = pickle.load(f)
    
train_features = []
train_labels = []
test_features = []
test_labels = []

train_ratio = 0.8

def process_data(sub_datas, train_ratio):
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    
    train_num = int(len(sub_datas) * train_ratio)
    # 注意需要打乱顺序，确定训练集和测试集
    train_indices = np.random.choice(len(sub_datas), train_num, replace=False)
    
    for i, data in tqdm(enumerate(sub_datas)):
        feature, label = extract_feature_target(data)
        if i in train_indices:
            train_features.append(feature)
            train_labels.append(label)
        else:
            test_features.append(feature)
            test_labels.append(label)
    
    return train_features, train_labels, test_features, test_labels

# 汇总所有的数据
train_features, train_labels = [], []
test_features, test_labels = [], []

with concurrent.futures.ProcessPoolExecutor() as executor:
    # 提交任务
    futures = {executor.submit(process_data, sub_datas, train_ratio): key for key, sub_datas in datas.items()}
    
    # 收集结果
    for future in concurrent.futures.as_completed(futures):
        t_features, t_labels, te_features, te_labels = future.result()
        train_features.extend(t_features)
        train_labels.extend(t_labels)
        test_features.extend(te_features)
        test_labels.extend(te_labels)

# 这里的 train_features, train_labels, test_features, and test_labels 包含所有的处理结果

# for _, sub_datas in datas.items():
#     train_num = int(len(sub_datas) * train_ratio)
#     # 注意需要打乱顺序，确定训练集和测试集
#     train_indices = np.random.choice(len(sub_datas), train_num, replace=False)
#     for i, data in enumerate(sub_datas):
#         feature, label = extract_feature_target(data)
#         if i in train_indices:
#             train_features.append(feature)
#             train_labels.append(label)
#         else:
#             test_features.append(feature)
#             test_labels.append(label)
        
        
    # node_type, num_inverted_predecessors, edge_index, nodes = feature
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graphs = []
    
for i in range(len(train_features)):
    node_type, num_inverted_predecessors, edge_index, nodes = train_features[i]
    x = torch.stack([node_type, num_inverted_predecessors], dim=1).float()
    data = Data(x=x, edge_index=edge_index, y=train_labels[i])
    graphs.append(data)
loader = DataLoader(graphs, batch_size=1, shuffle=True)

model = GCN(num_node_features=2, hidden_channels=16, num_output_features=1).to(device)  # 修改为实际的特征数和输出特征数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        target = data.y
        loss = F.mse_loss(out, target)  # 使用均方误差损失
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(loader):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        target = data.y
        loss = F.mse_loss(out, target)  # 使用均方误差损失
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

num_epochs = 200
for epoch in range(num_epochs):
    loss = train()
    train_loss = test(loader)  # 这里只使用训练数据进行评估
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Loss: {train_loss:.4f}')