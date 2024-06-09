import pickle
import numpy as np
import torch
import abc_py as abcPy
import os
from tqdm import tqdm

def get_feature(state):
    """
    从给定的电路状态文件中提取特征信息。
    
    参数:
    state (str): 电路状态文件的路径。
    
    返回:
    dict: 包含节点类型、反向前驱数量和边信息的字典。
    """
    _abc = abcPy.AbcInterface()
    _abc.start()
    _abc.read(state)
    
    data = {
        'node_type': np.zeros(_abc.numNodes(), dtype=int),
        'num_inverted_predecessors': np.zeros(_abc.numNodes(), dtype=int),
        'edge_index': [],
        'nodes': _abc.numNodes()
    }
    
    edge_src_index = []
    edge_target_index = []

    for nodeIdx in range(_abc.numNodes()):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2

        if nodeType == 4:
            data['num_inverted_predecessors'][nodeIdx] = 1
        elif nodeType == 5:
            data['num_inverted_predecessors'][nodeIdx] = 2

        if aigNode.hasFanin0():
            edge_src_index.append(nodeIdx)
            edge_target_index.append(aigNode.fanin0())
        
        if aigNode.hasFanin1():
            edge_src_index.append(nodeIdx)
            edge_target_index.append(aigNode.fanin1())

    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])

    return data

def write_aig(state):
    """
    将给定状态的电路转换为 AIG 格式文件，并提取其特征信息。
    
    参数:
    state (str): 电路状态描述。
    
    返回:
    dict: 包含节点类型、反向前驱数量和边信息的字典。
    """
    cache_path = f'./cache_task1/{state}.aig'
    
    if os.path.exists(cache_path):
        return get_feature(cache_path)
    
    lib_file = './lib/7nm/7nm.lib'
    synthesis_op_to_pos_dic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }

    circuit_name, actions = state.split('_')
    circuit_path = f'./InitialAIG/train/{circuit_name}.aig'
    next_state = f'./cache_task1/{state}.aig'
    
    if not actions:
        return get_feature(circuit_path)
    
    action_cmd = ' ; '.join(synthesis_op_to_pos_dic[int(action)] for action in actions)
    
    abc_run_cmd = (
        f'./yosys/yosys-abc -c "read {circuit_path} ; {action_cmd} ; '
        f'read_lib {lib_file} ; write {next_state}"'
    )
    os.system(abc_run_cmd)
    
    return get_feature(next_state)

def extract_feature_target(data):
    """
    从给定的数据中提取特征和目标值。
    
    参数:
    data (dict): 包含输入状态和目标值的字典。
    
    返回:
    tuple: 包含特征列表和目标张量的元组。
    """
    inputs, targets = data['input'], data['target']
    feature_list = [write_aig(state) for state in inputs]
    target_tensor = torch.tensor(targets, dtype=torch.float)

    return feature_list, target_tensor

def process_data(sub_datas, train_ratio, key):
    """
    处理单个子数据集，提取特征和标签，并划分训练集和测试集。
    
    参数:
    sub_datas (list): 子数据集。
    train_ratio (float): 训练集占比。
    key (str): 数据集名称。
    
    返回:
    tuple: 包含训练特征、训练标签、测试特征和测试标签的元组。
    """
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    
    train_num = int(len(sub_datas) * train_ratio)
    train_indices = np.random.choice(len(sub_datas), train_num, replace=False)
    
    for i, data in tqdm(enumerate(sub_datas)):
        feature, label = extract_feature_target(data)
        if i in train_indices:
            train_features.append(feature)
            train_labels.append(label)
        else:
            test_features.append(feature)
            test_labels.append(label)
    
    output_path = f'data/task1/{key}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump((train_features, train_labels, test_features, test_labels), f)
    
    return train_features, train_labels, test_features, test_labels