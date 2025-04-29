"""
    wikiv2.csv: user_id,item_id,timestamp,ori_time, state_label,comma_separated_list_of_features
                  the tiemstamp is the snapshot id  such as 0~12
                  the ori_time the original time from the original dataset # 1082448725
                  the link is undirected, so there are repeated links
    output:
    ml_wikiv2.csv: ,u,i,ts,label,idx
                ts is the original time 
                label is all 0s
                idx is the link id from 1... to the number of links
                there is no repeated links
"""
import pandas as pd
import numpy as np
import os

dataset_name = 'wikiv2'
data_name_ori = './pre/wikiv2_ori.csv'

data = pd.read_csv(data_name_ori)
print(len(set(list(data['user_id']))))
print(len(set(list(data['item_id']))))

print(data['user_id'].values.max())
print(data['item_id'].values.max())
# let's transform the data into the format of ml_uci.csv
data['label'] = 0

set1 = set(list(data['user_id'])+list(data['item_id']))

# delete the repeated links, remember to check whether it is repeated
# only keep the odd lines
data = data.iloc[::2,:]

train_node_set = set(list(data['user_id'])+list(data['item_id']))
print(len(train_node_set))
print(len(set(list(data['user_id']))))
print(len(set(list(data['item_id']))))
print(data['user_id'].values.max())
print(data['item_id'].values.max())
set2 = set(list(data['user_id'])+list(data['item_id']))


# each ori_time minus the min one
#data['ts'] = data['ori_time'] - min(data['ori_time'])
data['ts'] = data['ori_time']
# sort the data by the ts
data = data.sort_values(by='ts')

data = data[['user_id', 'item_id','ts', 'label',  'timestamp']]
# rename the columns
data.columns = ['u','i','ts','label', 'timestamp']

max_ts = data['timestamp'].values.max()

for timestamp in range(max_ts,max_ts+1):
    
    cur_data = data[data['timestamp']<=timestamp] #12
    train_data = cur_data[cur_data['timestamp']<timestamp-1] # 0 1 ... 10
    val_test_data = cur_data[cur_data['timestamp']>=timestamp-1] #11 12
    

    train_node_set = set(list(train_data['u'])+list(train_data['i']))
    print('len cur data: ', len(cur_data))
    # for the cur_data, delete the data [u,i] where u or i is not in the train_node_set
    new_cur_data = cur_data[cur_data['u'].isin(train_node_set) & cur_data['i'].isin(train_node_set)].copy()
    
    print('len newcur data, delete the new edges: ', len(new_cur_data))

    new_cur_data['idx'] = range(1,len(new_cur_data)+1)
    #new_cur_data.loc[:, 'idx'] = range(1, len(new_cur_data) + 1)

    new_cur_data.index = range(len(new_cur_data))

    save_path = os.path.join('./',str(timestamp))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    new_cur_data.to_csv(os.path.join(save_path, 'ml_'+dataset_name+'.csv'), index=True)

    # get the node features, the node features is np.zeros
    # the nodes are all the  u and i
    nodes = list(set(list(new_cur_data['u'])+list(new_cur_data['i'])))
    # np.zeros, save the node features to ml_uci_13_node.npy
    feat_dim = 172
    node_features = np.zeros((len(nodes),feat_dim))
    np.save(os.path.join(save_path,'ml_' + dataset_name + '_node.npy'), node_features)

    # get the edge features, the edge features is np.zeros
    # np.zeros, save the edge features to ml_uci_13.npy
    edge_features = np.zeros((len(new_cur_data),feat_dim))
    np.save(os.path.join(save_path,'ml_' + dataset_name + '.npy'), edge_features)







