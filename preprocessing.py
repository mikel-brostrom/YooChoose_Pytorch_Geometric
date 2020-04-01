import pandas as pd
import numpy as np
import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


clicks_df = pd.read_csv('./data/yoochoose-clicks.dat', header=None)
clicks_df.columns = ['session_id', 'timestamp', 'item_id', 'category']
print(clicks_df.head(5))

buy_df = pd.read_csv('./data/yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
print(buy_df.head(5))

item_encoder = LabelEncoder()
clicks_df['item_id'] = item_encoder.fit_transform(clicks_df.item_id)
print(clicks_df.head())

# randomly sample a couple of them
sampled_session_id = np.random.choice(clicks_df.session_id.unique(), 1000, replace=False)
clicks_df = clicks_df.loc[clicks_df.session_id.isin(sampled_session_id)]
print(clicks_df.nunique())

clicks_df['label'] = clicks_df.session_id.isin(buy_df.session_id)
print(clicks_df.head())

data_list = []

# process by session_id
grouped = clicks_df.groupby('session_id')
for session_id, group in grouped:
    sess_item_id = LabelEncoder().fit_transform(group.item_id)
    group = group.reset_index(drop=True)
    group['sess_item_id'] = sess_item_id
    node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values('sess_item_id').item_id.drop_duplicates().values

    node_features = torch.LongTensor(node_features).unsqueeze(1)
    target_nodes = group.sess_item_id.values[1:]
    source_nodes = group.sess_item_id.values[:-1]

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    x = node_features

    y = torch.FloatTensor([group.label.values[0]])

    data = Data(x=x, edge_index=edge_index, y=y)
    data_list.append(data)

data, slices = np.concatenate(data_list)
torch.save((data, slices), './data/processed.dataset')
