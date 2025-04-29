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
import os
import pickle
import torch
import utils as u
from scipy.sparse import csr_matrix, vstack

def load_edges_from_file(edges_file,folder,ids_str_to_int):
    edges = []
    not_found = 0

    file = edges_file
    
    file = os.path.join(folder,file)
    with open(file) as file:
        file = file.read().splitlines()

    cols = u.Namespace({'source': 0,
                        'target': 1,
                        'time': 3,
                        'label': 4})

    base_time = datetime.strptime("20140101", '%Y%m%d')
    #base_time = datetime.strptime("1980-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")


    
    for line in file[1:]:
        fields = line.split('\t')
        sr = fields[cols.source]
        tg = fields[cols.target]

        if sr in ids_str_to_int.keys() and tg in ids_str_to_int.keys():
            sr = ids_str_to_int[sr]
            tg = ids_str_to_int[tg]
            
            ts_ori = fields[cols.time]

            #time = fields[cols.time].split(' ')[0]
            #time = datetime.strptime(time,'%Y-%m-%d')
            #time = datetime.strptime(time,"%Y-%m-%d %H:%M:%S")
            #time = (time - base_time).days
            ts = int(time.mktime(datetime.strptime(ts_ori, "%Y-%m-%d %H:%M:%S").timetuple()))
            #time = int(time.mktime(datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timetuple()))

            label = int(fields[cols.label])
            #edges.append([sr,tg,ts,label])
            edges.append([sr,tg,ts,ts])
            #add the other edge to make it undirected
            #edges.append([tg,sr,ts,label])
            edges.append([tg,sr,ts,ts])
        else:
            not_found+=1

    return edges, not_found

file = '../web-redditEmbeddings-subreddits.csv'
with open(file) as file:
    file = file.read().splitlines()

ids_str_to_int = {}
id_counter = 0

feats = []

for line in file:
    line = line.split(',')
    #node id
    nd_id = line[0]
    if nd_id not in ids_str_to_int.keys():
        ids_str_to_int[nd_id] = id_counter
        id_counter += 1
        nd_feats = [float(r) for r in line[1:]]
        feats.append(nd_feats)
    else:
        print('duplicate id', nd_id)
        raise Exception('duplicate_id')

feats = torch.tensor(feats,dtype=torch.float)
num_nodes = feats.size(0)

edges = []
not_found = 0

#load edges in title
#edges_tmp, not_found_tmp = load_edges_from_file("soc-redditHyperlinks-title.tsv",
#                                                        "../",
#                                                        ids_str_to_int)
#edges.extend(edges_tmp)
#not_found += not_found_tmp

#load edges in bodies

edges_tmp, not_found_tmp = load_edges_from_file("soc-redditHyperlinks-body.tsv",
                                                        "../",
                                                        ids_str_to_int)
edges.extend(edges_tmp)
not_found += not_found_tmp
links_df = pd.DataFrame(edges)
links_df.columns = ['user_id', 'item_id','timestamp','ori_time']
links_df = links_df.sort_values(by=['ori_time'])

# group by source_subreddit and cheack the length of each group
group_len = []
for i, group in links_df.groupby('user_id'):
    #print(i, len(group))
    group_len.append(len(group))
# the statistics of the group length
pd.Series(group_len).describe()

# group_len rank topk
group_len2 = pd.Series(group_len)
group_len2.value_counts().sort_index(ascending=False).head(50)
# group_len distribution


print(len([i for i in group_len if i>10]) )# 10392
print(len([i for i in group_len if i>20])) # 6918
print(len([i for i in group_len if i>30])) # 5408
print(len([i for i in group_len if i>500]) )

print(len(links_df))
start = int(time.mktime(datetime.strptime("2016-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple()))
end = int(time.mktime(datetime.strptime("2016-12-25 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple()))
links_df = links_df[links_df['ori_time']>start]
links_df = links_df[links_df['ori_time']<end]

selected_sources = [i for i, group in links_df.groupby('user_id') if len(group) > 10 ] #and len(group) < 200
links_df = links_df[links_df['user_id'].isin(selected_sources)]

print(len([i for i in group_len if i>10]) )


links_df.to_csv('links_df.csv', index=False)

SLICE_DAYS = 30
START_DATE = links_df['timestamp'].min() #+ timedelta(240) # datetime.datetime(1993, 11, 30, 7, 0)
END_DATE = links_df['timestamp'].max() #- timedelta(12) # datetime.datetime(2002, 2, 28, 7, 0, 1)

slices_links = defaultdict(lambda : nx.MultiGraph())
slices_features = defaultdict(lambda : {})

print ("Start date", START_DATE)
print ("End date", END_DATE)

print('(END_DATE - START_DATE).days: ', (END_DATE - START_DATE)/(24*3600)) # 725 days


slice_id = 0
# Split the set of links in order by slices to create the graphs. 
#for (a, b, times, ori_time) in links:
for index, row in links_df.iterrows():
    a, b, times, ori_time = row['user_id'], row['item_id'], row['timestamp'], row['ori_time']

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
    print("Total # nodes", len(all_nodes), "max idx", max(all_nodes))
    
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
    
    # Remap nodes and edges for each slice
    for slice_id in slices_graph:
        G = nx.MultiGraph()
        for x in slices_graph[slice_id].nodes():
            G.add_node(node_idx[x])
        for x in slices_graph[slice_id].edges(data=True):
            G.add_edge(node_idx[x[0]], node_idx[x[1]], date=x[2]['date'])
        assert len(G.nodes()) == len(slices_graph[slice_id].nodes())
        assert len(G.edges()) == len(slices_graph[slice_id].edges())
        slices_graph_remap.append(G)
    
    # Remap features for each slice
    for slice_id in slices_features:
        features_remap = []
        slice_id = int(slice_id)
        for x in slices_graph_remap[slice_id].nodes():
            feature_vector = slices_features[slice_id][idx_node[x]]
            
            # Ensure compatibility (convert sparse matrix row to CSR if needed)
            if not isinstance(feature_vector, csr_matrix):
                feature_vector = csr_matrix(feature_vector)
            
            features_remap.append(feature_vector)
        
        # Use vstack to combine sparse rows into a single sparse matrix
        features_remap = vstack(features_remap)
        slices_features_remap.append(features_remap)
    
    return slices_graph_remap, slices_features_remap, node_idx, idx_node

slices_links_remap, slices_features_remap, node_idx, idx_node = remap(slices_links, slices_features)

#np.savez('graphs.npz', graph=slices_links_remap)
#np.savez('features.npz', feats=slices_features_remap)
final_remap = {v: k for k, v in node_idx.items()}
#remap_ori = {v: k for k, v in ids_str_to_int.items()}
node_features = []
for idx in range(len(node_idx)):
    #ori_idx = remap_ori[final_remap[idx]]
    ori_idx = final_remap[idx]
    #node_features.append(document_features[ori_idx])
    node_features.append(feats[ori_idx])
node_features = np.array(node_features)
print ("Node features shape", node_features.shape)
# save node_features
np.save("node_features.npy", node_features)

import pickle
with open('graphs.pkl', 'wb') as f:
    pickle.dump(slices_links_remap, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(slices_features_remap, f)

#with open('graphs.pkl', 'rb') as f:
#    slices_links_remap = pickle.load(f)     
#graphs = np.load("graphs.npz", allow_pickle=True)['graph']

graphs = slices_links_remap
import networkx as nx

with open('reddit_ori.csv', 'w') as f:
    # Write the header
    f.write('user_id,item_id,timestamp,ori_time,state_label,comma_separated_list_of_features\n')
    
    num_time = len(graphs)
    buffer = []  # Buffer to store lines before writing

    for timestamp in range(num_time):
        graph = nx.Graph(graphs[timestamp])  # Avoid recreating the graph repeatedly
        for (user, item) in graph.edges:
            ori_time = graph.edges[user, item]['date']
            ori_time = int(ori_time)
            user = int(user)
            item = int(item)
            timestamp = int(timestamp)
            
            # Add rows to the buffer
            buffer.append(f'{user},{item},{timestamp},{ori_time},0,0\n')
            buffer.append(f'{item},{user},{timestamp},{ori_time},0,0\n')

            # Write to the file in batches of 100,000 lines
            if len(buffer) >= 100000:
                f.writelines(buffer)
                buffer = []  # Clear the buffer

    # Write any remaining lines in the buffer
    if buffer:
        f.writelines(buffer)
# normalize the ori_time to be 0 to 1. use the min and max of the ori_time
# to normalize it
data = pd.read_csv('reddit_ori.csv')
data['ori_time'] = (data['ori_time'] - data['ori_time'].min()) / (data['ori_time'].max() - data['ori_time'].min())
data = data.sort_values(by=['ori_time'])
data.to_csv('reddit.csv', index=False)
# print the max time for each timestamp
for i, timestamp in data.groupby('timestamp'):
    print(i, timestamp['ori_time'].max())

