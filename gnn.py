# # cimport matplotlib.pyplot as plt
#import osmnx as ox
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


#from data_loader import YooChooseDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, SAGEConv, SGConv, SplineConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

np.random.seed(42)

datatset_size = 10000

clicks_df = pd.read_csv('./data/yoochoose-clicks.dat', header=None)
clicks_df.columns = ['session_id', 'timestamp', 'item_id', 'category']
print(clicks_df.head(5))

buy_df = pd.read_csv('./data/yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
print(buy_df.head(5))

item_encoder = LabelEncoder()
clicks_df['item_id'] = item_encoder.fit_transform(clicks_df.item_id)
print(clicks_df.head())

#randomly sample a couple of them
sampled_session_id = np.random.choice(clicks_df.session_id.unique(), datatset_size, replace=False)
clicks_df = clicks_df.loc[clicks_df.session_id.isin(sampled_session_id)]
print(clicks_df.nunique())

clicks_df['label'] = clicks_df.session_id.isin(buy_df.session_id)
print(clicks_df.head())

print(clicks_df.item_id.max() + 1)


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load('./data/processed.dataset')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['../input/yoochoose_click_binary_1M_sess.dataset']

    def download(self):
        pass

    def process(self):

        data_list = []

        # process by session_id
        grouped = clicks_df.groupby('session_id')
        for session_id, group in tqdm(grouped):
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

        data, slices = self.collate(data_list)
        torch.save((data, slices), './data/processed.dataset')



dataset = YooChooseBinaryDataset('./')
one_tenth_length = int(len(dataset) * 0.1)
dataset = dataset.shuffle()
train_dataset = dataset[:one_tenth_length * 8]
val_dataset = dataset[one_tenth_length * 8:one_tenth_length * 9]
test_dataset = dataset[one_tenth_length * 9:]
print(len(train_dataset), len(val_dataset), len(test_dataset))

embed_dim = 128
different_ids = 52707


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=different_ids, embedding_dim=embed_dim)
        #self.item_embedding = torch.nn.Embedding(num_embeddings=clicks_df.item_id.max() + 1, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x


def train(loader):
    model.train()

    loss_all = 0
    num_epochs = 30
    for epoch in range(num_epochs):
        print(epoch)
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            label = data.y.to(device)
            loss = crit(output, label)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
            #print(loss)
    return loss_all / len(train_dataset)


device = torch.device('cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.BCELoss()
train_loader = DataLoader(train_dataset, batch_size=512)
test_loader = DataLoader(test_dataset, batch_size=512)
val_loader = DataLoader(val_dataset, batch_size=512)




train(train_loader)


def evaluate(loader):
    model.eval()

    predictions = []
    labels = []
    print(len(loader))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    #if len(loader) == 0:
    #     return
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return roc_auc_score(labels, predictions)


for epoch in range(1):
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    test_acc = evaluate(test_loader)
    print(epoch, train_acc, val_acc, test_acc)
