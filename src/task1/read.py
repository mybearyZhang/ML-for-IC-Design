import pickle
import os

if __name__ == '__main__':
    files = os.listdir('./project_data')
    datas = {}
    for file in files:
        circuitName = file.split('_')[0]
        
        with open('./project_data/' + file, 'rb') as f:
            if circuitName not in datas:
                datas[circuitName] = [pickle.load(f)]
            else:
                datas[circuitName].append(pickle.load(f))
            
    with open('datas.pkl', 'wb') as f:
        pickle.dump(datas, f)