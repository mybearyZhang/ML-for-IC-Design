import pickle
import numpy as np
import torch
import abc_py as abcPy
import os
import re

# 读取project_data里的state
files = os.listdir('./project_data')
datas = {}
for file in files:
    # 每个文件都是pkl
    # 每个文件的名称是xx_num.pkl
    # xx是电路名，num是编号
    # 现在我要读取这些文件，并把相同电路名的文件合并
    circuitName = file.split('_')[0]
    
    with open('./project_data/' + file, 'rb') as f:
        if circuitName not in datas:
            datas[circuitName] = [pickle.load(f)]
        else:
            datas[circuitName].append(pickle.load(f))
        
with open('datas.pkl', 'wb') as f:
    pickle.dump(datas, f)