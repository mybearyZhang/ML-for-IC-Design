import pickle
import numpy as np
from tqdm import tqdm
from utils.feature import extract_feature_target, get_feature
import time
from multiprocessing import Manager, Pool, Lock
import concurrent.futures
import os
from multiprocessing import Lock

lock = Lock()

st_time = time.time()

def process_sequence(key, sub_datas):
    buffer = []
    for seq in tqdm(sub_datas[:3000]):
        states = [get_feature(f'./cache_task1/{aig_name}.aig') for aig_name in seq['input'][1:]]
        for i in range(2, len(seq['target'])):
            reward = seq['target'][i] - seq['target'][i - 1]
            state = states[i - 2]
            next_state = states[i - 1]
            action = int(seq['input'][i][-1])
            not_done = 1 if i < len(seq['target']) - 1 else 0
            buffer.append((state, action, next_state, reward, not_done))
    with lock:
        with open(f'./buffer/replay_buffer_all.pkl', 'ab') as f:
            pickle.dump(buffer, f)
    return None

if __name__ == '__main__':
    with open('datas.pkl', 'rb') as f:
        datas = pickle.load(f)
    
    if os.path.exists('./buffer/replay_buffer_all.pkl'):
        os.remove('./buffer/replay_buffer_all.pkl')

    os.makedirs('./buffer', exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for key, sub_datas in datas.items():
            futures.append(executor.submit(process_sequence, key, sub_datas))
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


