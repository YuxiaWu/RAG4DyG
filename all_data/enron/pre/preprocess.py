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
#,u,r,i,ts,label,idx
import pickle
from scipy.sparse import csr_matrix, vstack
links_ori = pd.read_csv('../edge_list.csv')
# sort the links_df by ts
#links_ori = links_ori.sort_values(by=['timestamp'])
#links_ori['timestamp'] = pd.to_datetime(links_ori['ts'].apply(lambda x: datetime.fromtimestamp(x)))
max_user_id = links_ori['u'].values.max()
max_item_id = links_ori['i'].values.max()
# min
min_user_id = links_ori['u'].values.min()
min_item_id = links_ori['i'].values.min()

# len(set(links_ori['item_id'].values.tolist())) # 1000
# len(set(links_ori['user_id'].values.tolist())) # 8227
# len(set(links_ori['item_id'].values.tolist() + links_ori['user_id'].values.tolist() )) # 8227

print("max_user_id: ", max_user_id )
print("max_item_id: ", max_item_id  )
print("min_user_id: ", min_user_id )
print("min_item_id: ", min_item_id  )
# 1, 42711
'''
max_user_id:  42702
max_item_id:  42711
min_user_id:  1
min_item_id:  1

'''
#links_ori['item_id'] = links_ori['item_id'] + max_user_id + 1
#print("max_item_id: ", links_ori['item_id'].values.max())

links = []
ts = []
ctr = 0
node_cnt = 0
node_idx = {}
idx_node = []

# re index the u and i row by row
for i, row in links_ori.iterrows():
    u = int(row['u'])
    i = int(row['i'])
    timestamp = int(row['ts'])
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
    links.append((node_idx[u],node_idx[i], row['ts'], row['ts']))

print ("Min ts", min(ts), "max ts", max(ts))    
#print ("Total time span: {} days".format(((max(ts) - min(ts)))/(24*3600)))  # 31 days
print ("Total time span: {} days".format(((max(ts) - min(ts)))/(1)))  # 31 days

links_df = pd.DataFrame(links)
links_df.columns = ['user_id', 'item_id','timestamp','ori_time']

# for timestamp, each tiemstamp minus the first timestamp, then divide by the last timestamp minus the first timestamp
#links_df['ori_diff'] = (links_df['ori_time'] - links_df['ori_time'].min()) 

#links_df['diff_norm'] = (links_df['ori_time'] - links_df['ori_time'].min()) / (links_df['ori_time'].max() - links_df['ori_time'].min())
links_df.to_csv('links_df.csv', index=False)
# len links: 797907, 15%: 119686, 20%: 159581, 25%: 199476
SLICE_DAYS = 45 # 0-15, test >15
START_DATE = links_df['timestamp'].min() #+ timedelta(240) # datetime.datetime(1993, 11, 30, 7, 0)
END_DATE = links_df['timestamp'].max() #- timedelta(12) # datetime.datetime(2002, 2, 28, 7, 0, 1)

slices_links = defaultdict(lambda : nx.MultiGraph())
slices_features = defaultdict(lambda : {})

print ("Start date", START_DATE)
print ("End date", END_DATE)

#print('(END_DATE - START_DATE).days: ', (END_DATE - START_DATE)/(24*3600)) # 725 days
print('(END_DATE - START_DATE).days: ', (END_DATE - START_DATE)) # 1005 days

# Split the set of links into slices to create the graphs
slice_id = 0
for a, b, times, ori_time in links:
    prev_slice_id = slice_id
    if times < START_DATE:
        continue
    if times > END_DATE:
        break
    
    # Calculate days difference and determine the current slice
    days_diff = (times - START_DATE)
    slice_id = days_diff // SLICE_DAYS
    
    if slice_id >= 16:
        # Handle special case for slice_id >= 16
        if 16 not in slices_links:
            slices_links[16].add_nodes_from(slices_links[15].nodes(data=True))  # Inherit nodes from slice 15
        slices_links[16].add_edges_from([(a, b, {'date': ori_time})])
    else:
        # Initialize new slice if transitioning
        if slice_id == 1+prev_slice_id and slice_id > 0:
            slices_links[slice_id] = nx.MultiGraph()
            slices_links[slice_id].add_nodes_from(slices_links[slice_id-1].nodes(data=True))
            assert (len(slices_links[slice_id].edges()) ==0)
            #assert len(slices_links[slice_id].nodes()) >0

        if slice_id == 1+prev_slice_id and slice_id ==0:
            slices_links[slice_id] = nx.MultiGraph()
        # Add nodes and edges for the current slice
        slices_links[slice_id].add_edges_from([(a, b, {'date': ori_time})])



'''
slice_id = 0
# Split the set of links in order by slices to create the graphs. 
for (a, b, times, ori_time) in links:
    prev_slice_id = slice_id
    datetime_object = times
    if datetime_object < START_DATE:
        continue
    if datetime_object > END_DATE:
        break
        #days_diff = (END_DATE - START_DATE)/(24*3600)
        days_diff = END_DATE - START_DATE
    else:
        #days_diff = (datetime_object - START_DATE)/(24*3600)
        days_diff = datetime_object - START_DATE
        
    
    slice_id = days_diff // SLICE_DAYS
    if slice_id==16:
        # for slice_id>15, we add all the remaining links to the last slice
        slices_links[16] = nx.MultiGraph()
        slices_links[16].add_nodes_from(slices_links[15].nodes(data=True))
        assert (len(slices_links[16].edges()) ==0)
        #assert len(slices_links[slice_id].nodes()) >0
    if slice_id >=16:
        if a not in slices_links[16]:
            slices_links[16].add_node(a)
        if b not in slices_links[16]:
            slices_links[16].add_node(b)    
        slices_links[16].add_edge(a,b, date=ori_time)

    else:
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
'''


from scipy.sparse import identity

# Iterate over each slice in slices_links
for slice_id in slices_links:
    print('-----------------')
    print("# nodes in slice", slice_id, len(slices_links[slice_id].nodes()))
    print("# edges in slice", slice_id, len(slices_links[slice_id].edges()))

    # Get the number of nodes in the current slice
    num_nodes = len(slices_links[max(slices_links.keys())].nodes())
    
    # Initialize slices_features for the current slice with an empty dictionary
    slices_features[slice_id] = {}

    # Create a sparse identity matrix only for the current slice
    for idx, node in enumerate(slices_links[slice_id].nodes()):
        # Assign a one-hot encoded sparse vector for each node
        one_hot_vector = identity(num_nodes, format='csr')[idx]
        slices_features[slice_id][node] = one_hot_vector

    #print("Shape of one-hot vector for nodes in slice", slice_id, one_hot_vector.shape)




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

'''
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
'''
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

import networkx as nx

graphs = slices_links_remap

with open('enron_ori.csv', 'w') as f:
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
data = pd.read_csv('enron_ori.csv')
data['ori_time'] = (data['ori_time'] - data['ori_time'].min()) / (data['ori_time'].max() - data['ori_time'].min())
data = data.sort_values(by=['ori_time'])
data.to_csv('enron.csv', index=False)
# print the max time for each timestamp
for i, timestamp in data.groupby('timestamp'):
    print(i, timestamp['ori_time'].max())