# YooChoose_Pytorch_Geometric

## The task

Given a sequence of click events performed by some user during a typical session in an e-commerce website, the goal is to predict whether the user is going to buy something or not, and if he is buying, what would be the items he is going to buy. The task could therefore be divided into two sub goals:

* Is the user going to buy items in this session? Yes | No
* If yes, what are the items that are going to be bought?

In this repository we will go for the first one

## The approach

Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks. The basic idea behind node embedding approaches is to use dimensionality reduction techniques to distill the high-dimensional information about a node’s graph neighborhood into a dense vector embedding. Previous research works have focused on embedding nodes from a single fixed graph, and many real-world applications require embeddings to be quickly generated for unseen nodes, or entirely new (sub)graphs. Hence, the idea is to use GraphSAGE (SAmple and aggreGatE). It uses node features (e.g., text attributes, node profile information, node degrees) in order to learn an embedding function that generalizes to unseen nodes

## The data

The training data comprises two different files:

    yoochoose-clicks.dat - Click events. Each record/line in the file has the following fields:
        Session ID – the id of the session. In one session there are one or many clicks.
        Timestamp – the time when the click occurred.
        Item ID – the unique identifier of the item.
        Category – the category of the item.
    yoochoose-buys.dat - Buy events. Each record/line in the file has the following fields:
        Session ID - the id of the session. In one session there are one or many buying events.
        Timestamp - the time when the buy occurred.
        Item ID – the unique identifier of item.
        Price – the price of the item.
        Quantity – how many of this item were bought.

The Session ID in yoochoose-buys.dat will always exist in the yoochoose-clicks.dat file – the records with the same Session ID together form the sequence of click events of a certain user during the session. The session could be short (few minutes) or very long (few hours), it could have one click or hundreds of clicks. All depends on the activity of the user. Download it from:

https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z

* Create a folder called `data`, in the root map
* Place all the files there

## The preprocessing

we subsample and then preprocess this portion of the dataset. item_ids are categorically encoded to ensure the encoded item_ids, which will later be mapped to an embedding matrix, starts at 0.

we treat each item in a session as a node, and therefore all items in the same session form a graph. To build the dataset, we group the preprocessed data by session_id and iterate over these groups. In each iteration, the item_id in each group are categorically encoded again since for each graph, the node index should count from 0.

`preprocess.py` preprocesses the files in the data folder:

```bash
python3 preprocess.py
```

## Requirements

Python 3.7 or later with 

- `np`
- `torch-scatter`
- `torch-sparse`
- `torch-cluster`
- `torch-spline-conv `
- `torch-geometric`

make sure that they work with your PyTorch version

## Training & Evaluation

`train.py` runs the training and evaluation:
```bash
python3 train.py
```
