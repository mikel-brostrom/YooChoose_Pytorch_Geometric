import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def test(test_loader, model, device):
    model.eval()

    predictions = []
    labels = []
    print(len(test_loader))
    nb = len(test_loader)

    # Disable gradients
    pbar = tqdm(enumerate(test_loader), total=nb)
    for batch_idx, data in pbar:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()

            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    #predictions = binarize(predictions, threshold=0.5, copy=True)
    #labels = binarize(labels, threshold=0.5, copy=True)



    return roc_auc_score(labels, predictions)
