import os
import numpy as np
import heapq
import torch
from torch_geometric.data import Data, DataLoader
from utils.aig import score_aig_baseline, score_aig_adp, score_aig_regularized, generate_next_aig
from utils.feature import get_feature

MAX_STEP = 10

class GreedySolver:
    def __init__(self, model, device, data_root='data/task2'):
        self._model = model
        self._device = device
        self._data_root = data_root

    def solve(self, aig_name):
        self._aig_name = aig_name

        os.makedirs(os.path.join(self._data_root, aig_name), exist_ok=True)

        state = f'{aig_name}_'
        self._baseline = score_aig_baseline(self._state_to_aig(state))
        for _ in range(MAX_STEP):
            heurs = []
            for action in range(7):
                new_state = state + str(action)
                child_file = self._state_to_aig(new_state)
                generate_next_aig(self._state_to_aig(state), action, child_file)
                heurs.append(self._heuristic(child_file))
            action = np.argmax(heurs)
            state = state + str(action)
        
        adp = score_aig_adp(self._state_to_aig(state))
        return adp, 1 - adp / self._baseline, state
    
    def _heuristic(self, aig_file):
        data = get_feature(aig_file)
        node_type, num_inverted_predecessors, edge_index, nodes = data.values()

        node_type_tensor = node_type.view(-1, 1).to(torch.float)
        num_inverted_predecessors_tensor = num_inverted_predecessors.view(-1, 1).to(torch.float)
        x = torch.cat([node_type_tensor, num_inverted_predecessors_tensor], dim=1)
        edge_index_tensor = edge_index.to(torch.long)

        graph_data = Data(x=x, edge_index=edge_index_tensor).to(self._device)

        out = self._model(graph_data).item()

        return (1 - out) * self._baseline
    
    def _state_to_aig(self, state):
        if state.endswith('_'):
            return os.path.join(self._data_root, f'{self._aig_name}.aig')
        else:
            return os.path.join(self._data_root, self._aig_name, f'{state}.aig')
