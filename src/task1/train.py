import torch
import pickle
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from ..utils.model import GCN
from tqdm import tqdm
import os
from time import time
import gc
import wandb
import argparse

def load_data(data_dir, max_train_samples=2500):
    start_time = time()
    graphs = []
    test_graphs = []

    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(data_dir, file), 'rb') as f:
                key = file.split('.')[0]
                train_feature, train_label, test_feature, test_label = pickle.load(f)
                print(f'{key} train: {len(train_feature)}, test: {len(test_feature)}, time: {time() - start_time}')
            
            for i in range(min(len(train_feature), max_train_samples)):
                for j in range(len(train_feature[i])):
                    graph_data = create_graph_data(train_feature[i][j], train_label[i][j])
                    graphs.append(graph_data)
            
            for i in range(len(test_feature)):
                for j in range(len(test_feature[i])):
                    graph_data = create_graph_data(test_feature[i][j], test_label[i][j])
                    test_graphs.append(graph_data)
            
            print(f'{key}, time: {time() - start_time}')
    
    return graphs, test_graphs

def create_graph_data(data, label):
    node_type, num_inverted_predecessors, edge_index, nodes = data.values()
    node_type_tensor = node_type.view(-1, 1).to(torch.float)
    num_inverted_predecessors_tensor = num_inverted_predecessors.view(-1, 1).to(torch.float)
    x = torch.cat([node_type_tensor, num_inverted_predecessors_tensor], dim=1)
    edge_index_tensor = edge_index.to(torch.long)
    return Data(x=x, edge_index=edge_index_tensor, y=torch.tensor([label]))

def get_data_loaders(train_graphs, test_graphs, batch_size=512, num_workers=4):
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def train_model(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).view(-1)
        target = data.y
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data).view(-1)
        target = data.y
        loss = F.mse_loss(out, target)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_graphs, test_graphs = load_data(args.data_dir, args.max_train_samples)
    train_loader, test_loader = get_data_loaders(train_graphs, test_graphs, args.batch_size, args.num_workers)
    
    del train_graphs, test_graphs
    gc.collect()
    
    model = GCN(num_node_features=2, hidden_channels=args.hidden_channels, num_output_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    wandb.init(project=args.project_name)
    
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss = evaluate_model(model, test_loader, device)

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.best_model_path)
            print(f'New best model saved at epoch {epoch:03d} with validation loss: {val_loss:.4f}')

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GCN model on IC design data.')
    parser.add_argument('--data_dir', type=str, default='data/task1/', help='Directory containing the data files.')
    parser.add_argument('--max_train_samples', type=int, default=2500, help='Maximum number of training samples to load.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Number of hidden channels in the GCN.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=5e-8, help='Weight decay for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--best_model_path', type=str, default='best_model_task1.pth', help='Path to save the best model.')
    parser.add_argument('--project_name', type=str, default='GNN-IC-Design', help='Project name for wandb logging.')
    
    args = parser.parse_args()
    main(args)
