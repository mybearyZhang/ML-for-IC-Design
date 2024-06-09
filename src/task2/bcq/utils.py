import numpy as np
import torch
import pickle
import torch_geometric
from torch_geometric.data import Data, DataLoader
import torch_geometric.data
import random
import abc_py as abcPy


class ReplayBuffer(object):
    def __init__(self, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.data_list = []

    def load(self, file_path):
        buffer = []
        with open(file_path, 'rb') as f:
            try:
                while True:
                    buffer_tmp = pickle.load(f)
                    buffer.extend(buffer_tmp)
            except EOFError:
                print("Finish reading.")
        self.data_list = []
        for transition in buffer:
            state, action, next_state, reward, not_done = transition
            reward = 0.025 - reward

            def extract_state(state):
                node_type, num_inverted_predecessors, edge_index, nodes = state.values()
                node_type_tensor = node_type.view(-1, 1).to(torch.float)
                num_inverted_predecessors_tensor = num_inverted_predecessors.view(-1, 1).to(torch.float)
                x = torch.cat([node_type_tensor, num_inverted_predecessors_tensor], dim=1)
                edge_index_tensor = edge_index.to(torch.long)
                return Data(x=x, edge_index=edge_index_tensor)

            state_graph = extract_state(state)
            next_state_graph = extract_state(next_state)
            self.data_list.append((state_graph, action, next_state_graph, reward, not_done))
        self.data_list = self.data_list[:self.max_size]

        print(f"Replay Buffer loaded with {len(self.data_list)} elements.")

    def sample(self):
        batch = random.sample(self.data_list, self.batch_size)
        state, action, next_state, reward, not_done = map(list, zip(*batch))
        
        # Assuming state and next_state are of type torch_geometric.data.Data
        states = self._collate(state).to(self.device)
        next_states = self._collate(next_state).to(self.device)
        actions = torch.tensor(action, dtype=torch.long).to(self.device).reshape(-1, 1)
        rewards = torch.tensor(reward, dtype=torch.float32).to(self.device).reshape(-1, 1)
        not_dones = torch.tensor(not_done, dtype=torch.float32).to(self.device).reshape(-1, 1)
        
        return states, actions, next_states, rewards, not_dones

    def _collate(self, graphs):
        batch = torch_geometric.data.Batch.from_data_list(graphs)
        return batch

    def __len__(self):
        return len(self.data_list)
