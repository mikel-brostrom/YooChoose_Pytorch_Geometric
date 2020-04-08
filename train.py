import os
import torch
from dataloader import LoadData
from torch_geometric.data import DataLoader
import argparse
from model import Net
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from test import test
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter


wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'


def train(args, model, device, train_loader, test_loader, val_loader, optimizer, save=True):
    # tb_writer = SummaryWriter()
    crit = torch.nn.BCELoss()

    nb = train_loader.__len__()

    best_roc_auc_score = 0
    for epoch in range(1, args.epochs + 1):

        # ----------------------------------------------------------------
        # start epoch ----------------------------------------------------
        # ----------------------------------------------------------------

        print('\tEPOCH', epoch, '/', args.epochs)

        model.train()
        pbar = tqdm(enumerate(train_loader), total=nb)

        for batch_idx, data in pbar:

            # ----------------------------------------------------------------
            # start batch ----------------------------------------------------
            # ----------------------------------------------------------------

            data = data.to(device)
            optimizer.zero_grad()  # zero the gradient buffers
            output = model(data)
            label = data.y.to(device)
            loss = crit(output, label)
            # loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()  # Does the update

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            # ----------------------------------------------------------------
            # end batch ------------------------------------------------------
            # ----------------------------------------------------------------

        # Update scheduler
        # scheduler.step()

        # Write Tensorboard results
        # if tb_writer:
        #     x = [loss]
        #     titles = ['MSE']
        #     for xi, title in zip(x, titles):
        #         tb_writer.add_scalar(title, xi, epoch)

        # Update best roc_auc_score
        roc_auc_score = test(test_loader, model, device)
        roc_auc_score2 = test(val_loader, model, device)
        print('test roc auc score:', roc_auc_score, "val", roc_auc_score2)

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write('%f' % roc_auc_score + '\n')

        if roc_auc_score > best_roc_auc_score:
            best_roc_auc_score = roc_auc_score

        # Save training results
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_roc_auc_score,
                         'training_results': f.read()}

            # Save last checkpoint
            torch.save(chkpt, last)

            # Save best checkpoint
            if round(best_roc_auc_score, 2) == round(roc_auc_score, 2):
                torch.save(chkpt, best)

            # Delete checkpoint
            del chkpt

    # ----------------------------------------------------------------
    # end epoch ------------------------------------------------------
    # ----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='PyTorch Battery')

    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    print('\n\n\tTrain...\n')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # data = load_preprocessed_data()
    batch_size = 512

    dataset = LoadData('./data/processed.dataset')
    # dataset = torch.load('./data/processed.dataset')

    one_tenth_length = int(len(dataset) * 0.1)
    train_dataset = dataset[:one_tenth_length * 8]
    val_dataset = dataset[one_tenth_length * 8:one_tenth_length * 9]
    test_dataset = dataset[one_tenth_length * 9:]

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=1,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=1,
                            shuffle=True,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=1,
                             shuffle=True,
                             pin_memory=True)

    train(args, model, device, train_loader, test_loader, val_loader, optimizer)


if __name__ == "__main__":
    main()
