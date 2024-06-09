import os
import numpy as np
import heapq
import torch
from torch_geometric.data import Data, DataLoader
from utils.aig import score_aig_baseline, score_aig_adp, score_aig_regularized, generate_next_aig
from utils.feature import get_feature

MAX_STEP = 10

class AStarSolver:
    def __init__(self, model, device, data_root='data/task2'):
        self._model = model
        self._device = device
        self._data_root = data_root

    def solve(self, aig_name):
        self._aig_name = aig_name
        self._best_adp = float('inf')
        self._best_state = None

        self._visited_states = set()
        node_queue = []

        os.makedirs(os.path.join(self._data_root, aig_name), exist_ok=True)

        init_state = f'{aig_name}_'
        init_aig_file = self._state_to_aig(init_state)
        self._baseline = score_aig_baseline(init_aig_file)
        heapq.heappush(node_queue, Node(init_state, score_aig_adp(init_aig_file), self._heuristic(init_aig_file), None))

        while node_queue:
            cur_node = heapq.heappop(node_queue)

            if cur_node.cost < self._best_adp:
                self._best_adp = cur_node.cost
                self._best_state = cur_node.state
            print(cur_node.state, self._best_adp, self._baseline)

            if tuple(cur_node.state) in self._visited_states:
                continue
            # optional pruning
            # if cur_node.eval_cost > self._best_adp:
            #     continue

            # finish searching
            if self._finish(cur_node.state):
                continue
            
            self._visited_states.add(tuple(cur_node.state))

            # explore new nodes
            for node in self._explore(cur_node):
                heapq.heappush(node_queue, node)

        return self._best_adp, 1 - self._best_adp / self._baseline, self._best_state

    def _explore(self, cur_node):
        explore_nodes = []
        for action in range(7):
            new_state = cur_node.state + str(action)
            child_file = self._state_to_aig(new_state)
            generate_next_aig(self._state_to_aig(cur_node.state), action, child_file)
            past_adp = score_aig_adp(child_file)
            future_adp = self._heuristic(child_file)
            explore_nodes.append(Node(new_state, past_adp, future_adp, cur_node))
        if not cur_node.state.endswith('_'):
            os.remove(self._state_to_aig(cur_node.state))
        return explore_nodes
    
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

    def _finish(self, state):
         _, actions = state.split('_')
         return len(actions) >= MAX_STEP
    
    def _state_to_aig(self, state):
        if state.endswith('_'):
            return os.path.join(self._data_root, f'{self._aig_name}.aig')
        else:
            return os.path.join(self._data_root, self._aig_name, f'{state}.aig')

class Node(object):
    def __init__(self, state, g, h, parent):
        self.state = state
        self._g = g
        # self._f = g + h
        self._h = h
        self.parent = parent

    @property
    def cost(self):
        return self._g
    
    @property
    def eval_cost(self):
        return self._g + self._h

    def __lt__(self, other):
        return self.eval_cost < other.eval_cost