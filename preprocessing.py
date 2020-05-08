import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


datatset_size = 10000

# yoochoose-clicks.dat - Click events. Each record/line in the file has the following fields:
#     Session ID – the id of the session. In one session there are one or many clicks.
#     Timestamp – the time when the click occurred.
#     Item ID – the unique identifier of the item.
#     Category – the category of the item.
clicks_df = pd.read_csv('./data/yoochoose-clicks.dat', header=None)
clicks_df.columns = ['session_id', 'timestamp', 'item_id', 'category']

# filter out item session with length < 2
clicks_df['valid_session'] = clicks_df.session_id.map(clicks_df.groupby('session_id')['item_id'].size() > 2)
clicks_df = clicks_df.loc[clicks_df.valid_session].drop('valid_session',axis=1)
print(clicks_df.nunique())
print(clicks_df.head(5))

# yoochoose-buys.dat - Buy events. Each record/line in the file has the following fields:
#     Session ID - the id of the session. In one session there are one or many buying events.
#     Timestamp - the time when the buy occurred.
#     Item ID – the unique identifier of item.
#     Price – the price of the item.
#     Quantity – how many of this item were bought.
buy_df = pd.read_csv('./data/yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
print(buy_df.head())

# Encode item_id with values between 0 and n_classes - 1 (for embedding)
item_encoder = LabelEncoder()
clicks_df['item_id'] = item_encoder.fit_transform(clicks_df.item_id)
print(min(clicks_df['item_id']))
print(max(clicks_df['item_id']))
print(clicks_df.head())

# randomly sample a couple of them
sampled_session_id = np.random.choice(clicks_df.session_id.unique(), datatset_size, replace=False)
clicks_df = clicks_df.loc[clicks_df.session_id.isin(sampled_session_id)]
print(clicks_df.nunique())

# average length of session 
print(clicks_df.groupby('session_id')['item_id'].size().mean())

# add a boolean column named label to the clicks_df representing wether the click in the session is
# buy or not
clicks_df['label'] = clicks_df.session_id.isin(buy_df.session_id)

print(clicks_df.label.value_counts())
print(clicks_df.label.unique())
print(clicks_df.item_id.max() + 1)

data_list = []

# process by session_id
grouped = clicks_df.groupby('session_id')
for session_id, group in tqdm(grouped):
    sess_item_id = LabelEncoder().fit_transform(group.item_id)
    group = group.reset_index(drop=True)
    group['sess_item_id'] = sess_item_id
    node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id', 'timestamp']].sort_values('sess_item_id').item_id.drop_duplicates().values

    node_features = torch.LongTensor(node_features).unsqueeze(1)
    target_nodes = group.sess_item_id.values[1:]
    source_nodes = group.sess_item_id.values[:-1]

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    x = node_features

    y = torch.FloatTensor([group.label.values[0]])

    data = Data(x=x, edge_index=edge_index, y=y)
    data_list.append(data)

torch.save(data_list, './data/processed.dataset')
