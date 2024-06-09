import pickle
import numpy as np
from tqdm import tqdm
from utils.feature import extract_feature_target
import concurrent

with open('datas2.pkl', 'rb') as f:
    datas = pickle.load(f)
    
train_features = []
train_labels = []
test_features = []
test_labels = []

train_ratio = 0.8

def process_data(sub_datas, train_ratio, key):
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    
    train_num = int(len(sub_datas) * train_ratio)
    train_indices = np.random.choice(len(sub_datas), train_num, replace=False)
    
    for i, data in enumerate(tqdm(sub_datas)):
        feature, label = extract_feature_target(data)
        if i in train_indices:
            train_features.append(feature)
            train_labels.append(label)
        else:
            test_features.append(feature)
            test_labels.append(label)
            
    with open(f'./task2/{key}.pkl', 'wb') as f:
        pickle.dump((train_features, train_labels, test_features, test_labels), f)


with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_data, sub_datas, train_ratio, key): key for key, sub_datas in datas.items()}
    
    for future in concurrent.futures.as_completed(futures):
        future.result()