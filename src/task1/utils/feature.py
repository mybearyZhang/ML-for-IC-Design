import pickle
import numpy as np
import torch
import abc_py as abcPy
import os
import re

def get_feature(state):
    # 初始化 ABC 接口
    _abc = abcPy.AbcInterface()
    _abc.start()

    # 读取电路状态文件
    _abc.read(state)

    # 初始化数据字典
    data = {}

    # 获取节点数
    numNodes = _abc.numNodes()

    # 初始化数据数组
    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []

    # 遍历每个节点并填充数据
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0

        # 确定节点类型
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2

        # 检查是否有反向前驱
        if nodeType == 4:
            data['num_inverted_predecessors'][nodeIdx] = 1
        if nodeType == 5:
            data['num_inverted_predecessors'][nodeIdx] = 2

        # 检查是否有fanin 0
        if aigNode.hasFanin0():
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)

        # 检查是否有fanin 1
        if aigNode.hasFanin1():
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)

    # 将列表转换为torch张量并添加到数据字典中
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['nodes'] = numNodes
    return data

def write_aig(state):
    if os.path.exists('./cache_task1/' + state + '.aig'):
        return get_feature('./cache_task1/' + state + '.aig')
    
    libFile = './lib/7nm/7nm.lib'
    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }

    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    nextState = './cache_task1/' + state + '.aig'  # current AIG file
    actionCmd = ''
    if actions == '':
        feature = get_feature(circuitPath)
    else:
        for action in actions:
            actionCmd += synthesisOpToPosDic[int(action)] + ' ; '

        abcRunCmd = (
            './yosys/yosys-abc -c "read ' + circuitPath + ' ; ' +
            actionCmd +
            ' read_lib ' + libFile + ' ; write ' + nextState + '"'
        )
        os.system(abcRunCmd)

        feature = get_feature(nextState)
    return feature

def extract_feature_target(data):
    inputs, targets = data['input'], data['target']
    feature_list, target_tensor = [], torch.zeros(len(inputs))
    for i in range(len(inputs)):
        state = inputs[i]
        feature_list.append(write_aig(state))
        target_tensor[i] = (targets[i])
        
    return feature_list, target_tensor