import os
import re

# ========================================================================================================

'''
读了一个state，然后用ABC跑一个action，然后保存新的state

'''

state = 'alu2_0130622'
circuitName, actions = state.split('_')
circuitPath = './InitialAIG/train/' + circuitName + '.aig'
libFile = './lib/7nm/7nm.lib'
logFile = 'alu2.log'
nextState = state + '.aig'  # current AIG file
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}
actionCmd = ''
for action in actions:
    actionCmd += synthesisOpToPosDic[int(action)] + ' ; '
    
abcRunCmd = (
    './yosys/yosys-abc -c "read ' + circuitPath + ' ; ' +
    actionCmd +
    ' read_lib ' + libFile + ' ; write ' + nextState +
    ' ; print_stats" > ' + logFile
)
os.system(abcRunCmd)

# ========================================================================================================
'''
Evaluate the new state
'''

AIG = state + '.aig'

abcRunCmd = f"yosys-abc -c \"read {AIG}; read_lib {libFile}; map; topo; stime\" > {logFile}"

# Execute the command
os.system(abcRunCmd)

# Open the log file and extract area information
with open(logFile) as f:
    lines = f.readlines()
    areaInformation = re.findall(r'[a-zA-Z0-9.]+', lines[-1])

# Calculate the evaluation value
eval = float(areaInformation[-9]) * float(areaInformation[-4])

# Print the evaluation value
print("Evaluation value:", eval)

# ========================================================================================================
'''
Resynthesize the new state
'''

RESYN2_CMD = ("balance; rewrite; refactor; balance; rewrite; rewrite -z; "
              "balance; refactor -z; rewrite -z; balance;")

nextBench = state + '.bench'

# Construct the command to run yosys with abc
abcRunCmd = (f"yosys-abc -c \"read {circuitPath}; {RESYN2_CMD} "
             f"read_lib {libFile}; write {nextState}; write_bench -l {nextBench}; map; topo; stime\" > {logFile}")

# Execute the command
os.system(abcRunCmd)

# Open the log file and extract area information
with open(logFile) as f:
    lines = f.readlines()
    areaInformation = re.findall(r'[a-zA-Z0-9.]+', lines[-1])

# Calculate the baseline and evaluation value
baseline = float(areaInformation[-9]) * float(areaInformation[-4])
eval = 1 - (eval / baseline)

# Print the evaluation value
print("Evaluation value:", eval)

# ========================================================================================================

import numpy as np
import torch

import abc_py as abcPy

# 初始化 ABC 接口
_abc = abcPy.AbcInterface()
_abc.start()

# 读取电路状态文件
_abc.read(state + '.aig')

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

# 打印数据以进行调试
print(data)
