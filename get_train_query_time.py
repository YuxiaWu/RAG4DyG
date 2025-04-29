import pandas as pd
import os
import pandas as pd
import sys
import torch

def load_data(data_name, timestamp):
    file_path = os.path.join('resources', data_name, timestamp, f'ml_{data_name}.csv')
    data = pd.read_csv(file_path)
    data_reverse = data.copy()
    data_reverse['u'] = data['i']
    data_reverse['i'] = data['u']
    ml_data = pd.concat([data, data_reverse], ignore_index=True)
    ml_data = ml_data.sort_values(by=['ts'])
    return ml_data

def get_query_time(ml_data, queryID, timestamp):
    query_data = ml_data[ml_data['u'] == queryID]
    query_data = query_data[query_data['timestamp']<=int(timestamp)-2]
    max_timestamp = query_data[query_data['timestamp'] <= int(timestamp) - 2]['timestamp'].max()
    try:
        query_time = query_data[query_data['timestamp']<max_timestamp].iloc[-1]['ts']
    except:
        query_time = query_data[query_data['timestamp']==max_timestamp].iloc[-1]['ts']        
    return query_time

def get_query_time_all(data_name, ml_data, data_path):

    with open(data_path, 'r') as f:
        data = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    query_id_time_diff = {}
    all_query_time = []
    for seq in data:
        # for each sample 
        queryID = int(seq.split('<|history|>')[1].split(' ')[1])
        query_id_time_diff[queryID] = {}
        query_time = get_query_time(ml_data, queryID, timestamp)
        all_query_time.append(query_time/scales[data_name] )

    all_query_time_tensor = torch.tensor(all_query_time, dtype=torch.float)
    torch.save(all_query_time_tensor, os.path.join('resources', data_name+'_train_query_time.pt'))


if __name__ == '__main__':
    data_name = sys.argv[1]
    timestamp = sys.argv[2]
    scales = {
        'UCI_13': 3600*24,
        'hepth': 3600*24*30,
        'dialog': 1,
        'wikiv2':3600*24,
        'enron': 1,
        'reddit': 1
    }
    ml_data = load_data(data_name, timestamp)        
    train_data_path = os.path.join('resources', data_name, timestamp, 'train.link_prediction')
    get_query_time_all(data_name, ml_data, train_data_path)

