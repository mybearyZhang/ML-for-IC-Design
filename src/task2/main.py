from itertools import combinations
from tqdm import tqdm
import time
import argparse
import torch
from utils.model import GCN
from astar import AStarSolver
from greedy import GreedySolver
from rl import RLSolver


def get_model(model_path, device):
    model = GCN(num_node_features=2, hidden_channels=16, num_output_features=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aig', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='/root/ML-for-IC-Design/data/InitialAIG/test')
    parser.add_argument('--method', type=str, default='astar', choices=['astar', 'greedy', 'rl'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.method == 'astar':
        model = get_model(args.model_path, device)
        solver = AStarSolver(model, device, args.data_root)
    elif args.method == 'greedy':
        model = get_model(args.model_path, device)
        solver = GreedySolver(model, device, args.data_root)
    elif args.method == 'rl':
        solver = RLSolver(args.model_path, device, args.data_root)

    st_time = time.monotonic()
    result = solver.solve(args.aig)
    ed_time = time.monotonic()

    print('===========================')
    print('Method:', args.method, '| AIG:', f'{args.aig}')
    print(f'Time used: {ed_time - st_time:.2f}s')
    print('Min ADP:', result[0])
    print('Regularized ADP:', result[1])
    print('Actions:', result[2])
    print('===========================')

        