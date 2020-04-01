# YooChoose_Pytorch_Geometric

## The task

Given a sequence of click events performed by some user during a typical session in an e-commerce website, the goal is to predict whether the user is going to buy something or not, and if he is buying, what would be the items he is going to buy. The task could therefore be divided into two sub goals:

* Is the user going to buy items in this session? Yes|No
* If yes, what are the items that are going to be bought?

    In this repository we will go for the first one

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

The Session ID in yoochoose-buys.dat will always exist in the yoochoose-clicks.dat file – the records with the same Session ID together form the sequence of click events of a certain user during the session. The session could be short (few minutes) or very long (few hours), it could have one click or hundreds of clicks. All depends on the activity of the user. 
