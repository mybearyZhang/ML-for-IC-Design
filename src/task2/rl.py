import os
import numpy as np
import heapq
import torch
from torch_geometric.data import Data, DataLoader
from utils.aig import score_aig_baseline, score_aig_adp, generate_next_aig
from utils.feature import aig_to_data
from bcq.bcq import discrete_BCQ

MAX_STEP = 10

class RLSolver:
    def __init__(self, model_path, device, data_root='data/task2'):
        self._model = discrete_BCQ(7, device)
        self._model.load(model_path)
        self._device = device
        self._data_root = data_root

    def solve(self, aig_name):
        self._aig_name = aig_name

        os.makedirs(os.path.join(self._data_root, aig_name), exist_ok=True)

        state = f'{aig_name}_'
        self._baseline = score_aig_baseline(self._state_to_aig(state), logFile=f'{self._aig_name}.log')
        for _ in range(MAX_STEP):
            action = self._get_action(state)
            new_state = state + str(action)
            child_file = self._state_to_aig(new_state)
            generate_next_aig(self._state_to_aig(state), action, child_file)
            state = new_state
        
        adp = score_aig_adp(self._state_to_aig(state))
        return adp, 1 - adp / self._baseline, state
    
    def _get_action(self, state):
        data = aig_to_data(self._state_to_aig(state))
        action = self._model.select_action(data, eval=True)
        return action

    def _state_to_aig(self, state):
        if state.endswith('_'):
            return os.path.join(self._data_root, f'{self._aig_name}.aig')
        else:
            return os.path.join(self._data_root, self._aig_name, f'{state}.aig')
