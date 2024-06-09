import pickle
import numpy as np
from tqdm import tqdm
from ..utils.feature import extract_feature_target
import concurrent.futures

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def process_data(sub_datas, train_ratio, key):
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

def collect_data(datas, train_ratio):
    train_features, train_labels = [], []
    test_features, test_labels = [], []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_data, sub_datas, train_ratio, key): key for key, sub_datas in datas.items()}
        
        for future in concurrent.futures.as_completed(futures):
            t_features, t_labels, te_features, te_labels = future.result()
            train_features.extend(t_features)
            train_labels.extend(t_labels)
            test_features.extend(te_features)
            test_labels.extend(te_labels)
    
    return train_features, train_labels, test_features, test_labels

def main():
    data_file = 'datas2.pkl'
    train_ratio = 0.8
    
    datas = load_data(data_file)
    train_features, train_labels, test_features, test_labels = collect_data(datas, train_ratio)
    
    print(f'Train features: {len(train_features)}, Train labels: {len(train_labels)}')
    print(f'Test features: {len(test_features)}, Test labels: {len(test_labels)}')

if __name__ == '__main__':
    main()
