import dill
from collections import defaultdict
from datetime import datetime, timedelta

from scipy.sparse import csr_matrix

import networkx as nx
import numpy as np
import pandas as pd
import json
import time

# load pkl

import pickle

links_ori = pd.read_csv('../tgbl-wiki_edgelist_v2.csv')
# sort the links_df by ts
links_ori = links_ori.sort_values(by=['timestamp'])
#links_ori['timestamp'] = pd.to_datetime(links_ori['ts'].apply(lambda x: datetime.fromtimestamp(x)))
max_user_id = links_ori['user_id'].values.max()
max_item_id = links_ori['item_id'].values.max()

# len(set(links_ori['item_id'].values.tolist())) # 1000
# len(set(links_ori['user_id'].values.tolist())) # 8227
# len(set(links_ori['item_id'].values.tolist() + links_ori['user_id'].values.tolist() )) # 8227

print("max_user_id: ", max_user_id )
print("max_item_id: ", max_item_id  )

links_ori['item_id'] = links_ori['item_id'] + max_user_id + 1
print("max_item_id: ", links_ori['item_id'].values.max())

links = []
ts = []
ctr = 0
node_cnt = 0
node_idx = {}
idx_node = []

# re index the u and i row by row
for i, row in links_ori.iterrows():
    u = int(row['user_id'])
    i = int(row['item_id'])
    timestamp = int(row['timestamp'])
    ts.append(timestamp)
    ctr += 1
    if ctr % 100000 == 0:
        print (ctr)
    if u not in node_idx:
        node_idx[u] = node_cnt 
        node_cnt += 1
    if i not in node_idx:
        node_idx[i] = node_cnt 
        node_cnt += 1
    #links.append((node_idx[u],node_idx[i], row['timestamp'], row['ts']))
    links.append((node_idx[u],node_idx[i], row['timestamp'], row['timestamp']))

print ("Min ts", min(ts), "max ts", max(ts))    
print ("Total time span: {} days".format(((max(ts) - min(ts)))/(24*3600)))  # 31 days

links_df = pd.DataFrame(links)
links_df.columns = ['user_id', 'item_id','timestamp','ori_time']

# for timestamp, each tiemstamp minus the first timestamp, then divide by the last timestamp minus the first timestamp
#links_df['ori_diff'] = (links_df['ori_time'] - links_df['ori_time'].min()) 

#links_df['diff_norm'] = (links_df['ori_time'] - links_df['ori_time'].min()) / (links_df['ori_time'].max() - links_df['ori_time'].min())
links_df.to_csv('links_df.csv', index=False)

SLICE_DAYS = 2
START_DATE = links_df['timestamp'].min() #+ timedelta(240) # datetime.datetime(1993, 11, 30, 7, 0)
END_DATE = links_df['timestamp'].max() #- timedelta(12) # datetime.datetime(2002, 2, 28, 7, 0, 1)

slices_links = defaultdict(lambda : nx.MultiGraph())
slices_features = defaultdict(lambda : {})

print ("Start date", START_DATE)
print ("End date", END_DATE)

print('(END_DATE - START_DATE).days: ', (END_DATE - START_DATE)/(24*3600)) # 725 days

slice_id = 0
# Split the set of links in order by slices to create the graphs. 
for (a, b, times, ori_time) in links:
    prev_slice_id = slice_id
    datetime_object = times
    if datetime_object < START_DATE:
        continue
    if datetime_object > END_DATE:
        break
        days_diff = (END_DATE - START_DATE)/(24*3600)
    else:
        days_diff = (datetime_object - START_DATE)/(24*3600)
        
    
    slice_id = days_diff // SLICE_DAYS
    
    
    if slice_id == 1+prev_slice_id and slice_id > 0:
        slices_links[slice_id] = nx.MultiGraph()
        slices_links[slice_id].add_nodes_from(slices_links[slice_id-1].nodes(data=True))
        assert (len(slices_links[slice_id].edges()) ==0)
        #assert len(slices_links[slice_id].nodes()) >0

    if slice_id == 1+prev_slice_id and slice_id ==0:
        slices_links[slice_id] = nx.MultiGraph()

    if a not in slices_links[slice_id]:
        slices_links[slice_id].add_node(a)
    if b not in slices_links[slice_id]:
        slices_links[slice_id].add_node(b)    
    slices_links[slice_id].add_edge(a,b, date=ori_time)
    
for slice_id in slices_links:
    print ("# nodes in slice", slice_id, len(slices_links[slice_id].nodes()))
    print ("# edges in slice", slice_id, len(slices_links[slice_id].edges()))
    
    temp = np.identity(len(slices_links[max(slices_links.keys())].nodes()))
    print ("Shape of temp matrix", temp.shape)
    slices_features[slice_id] = {}
    for idx, node in enumerate(slices_links[slice_id].nodes()):
        slices_features[slice_id][node] = temp[idx]


def remap(slices_graph, slices_features):
    all_nodes = []
    for slice_id in slices_graph:
        assert len(slices_graph[slice_id].nodes()) == len(slices_features[slice_id])
        all_nodes.extend(slices_graph[slice_id].nodes())
    all_nodes = list(set(all_nodes))
    print ("Total # nodes", len(all_nodes), "max idx", max(all_nodes))
    ctr = 0
    node_idx = {}
    idx_node = []
    for slice_id in slices_graph:
        for node in slices_graph[slice_id].nodes():
            if node not in node_idx:
                node_idx[node] = ctr
                idx_node.append(node)
                ctr += 1
    slices_graph_remap = []
    slices_features_remap = []
    for slice_id in slices_graph:
        G = nx.MultiGraph()
        for x in slices_graph[slice_id].nodes():
            G.add_node(node_idx[x])
        for x in slices_graph[slice_id].edges(data=True):
            G.add_edge(node_idx[x[0]], node_idx[x[1]], date=x[2]['date'])
        assert (len(G.nodes()) == len(slices_graph[slice_id].nodes()))
        assert (len(G.edges()) == len(slices_graph[slice_id].edges()))
        slices_graph_remap.append(G)
    
    for slice_id in slices_features:
        features_remap = []
        slice_id = int(slice_id)
        for x in slices_graph_remap[slice_id].nodes():
            features_remap.append(slices_features[slice_id][idx_node[x]])
            #features_remap.append(np.array(slices_features[slice_id][idx_node[x]]).flatten())
        features_remap = csr_matrix(np.squeeze(np.array(features_remap)))
        slices_features_remap.append(features_remap)
    return (slices_graph_remap, slices_features_remap, node_idx, idx_node)

slices_links_remap, slices_features_remap, node_idx, idx_node = remap(slices_links, slices_features)

#np.savez('graphs.npz', graph=slices_links_remap)
#np.savez('features.npz', feats=slices_features_remap)

import pickle
with open('graphs.pkl', 'wb') as f:
    pickle.dump(slices_links_remap, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(slices_features_remap, f)

#with open('graphs.pkl', 'rb') as f:
#    slices_links_remap = pickle.load(f)     
#graphs = np.load("graphs.npz", allow_pickle=True)['graph']

graphs = slices_links_remap
with open('wikiv2_ori.csv', 'w') as f:
    f.write('user_id,item_id,timestamp,ori_time, state_label,comma_separated_list_of_features\n')
    
    num_time = len(graphs)    
    for timestamp in range(num_time):
        for (user, item) in nx.Graph(graphs[timestamp]).edges:
            ori_time = nx.Graph(graphs[timestamp]).edges[user, item]['date']
            # datetime.datetime(2004, 4, 30, 8, 23, 42)
            # let's convert it to a number
            #ori_time = ori_time.timestamp()
            #ori_time = int(ori_time.strftime("%Y%m%d%H%M%S"))


            ori_time = int(ori_time)
            user = int(user)
            item = int(item)
            timestamp = int(timestamp)
            f.write('%d,%d,%d,%d,0,0\n'%(user, item, timestamp, ori_time))
            f.write('%d,%d,%d,%d, 0,0\n'%(item, user, timestamp, ori_time))

# normalize the ori_time to be 0 to 1. use the min and max of the ori_time
# to normalize it
data = pd.read_csv('wikiv2_ori.csv')
data['ori_time'] = (data['ori_time'] - data['ori_time'].min()) / (data['ori_time'].max() - data['ori_time'].min())
data = data.sort_values(by=['ori_time'])
data.to_csv('wikiv2.csv', index=False)
# print the max time for each timestamp
for i, timestamp in data.groupby('timestamp'):
    print(i, timestamp['ori_time'].max())