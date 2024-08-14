import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, savepath=None, patience=7, verbose=False, delta=0, num_n_fold=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.min_loss = np.inf  # infinity
        # self.max_ci = -np.inf  # infinitesimal
        self.early_stop = False
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath

    def __call__(self, loss, ci, model, num_epoch):
        if self.min_loss == np.inf:
            self.save_checkpoint(loss, ci, model)
            self.min_loss = loss
        elif loss < self.min_loss:
            self.save_checkpoint(loss, ci, model)
            self.min_loss = loss
            self.counter = 0
        # elif loss < self.min_loss and ci <= self.max_ci:
        #     self.save_checkpoint(loss, ci, model)
        #     self.min_loss = loss
        #     self.counter = 0
        # elif loss >= self.min_loss and ci > self.max_ci:
        #     self.save_checkpoint(loss, ci, model)
        #     self.max_ci = ci
        #     self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


    def save_checkpoint(self, loss, ci , model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print("Have a new best checkpoint:  ",end='')
            print_msg = (f'valid_loss: {loss:.4f} ' +
                         f'valid_ci: {ci:.4f} ')
            print(print_msg,end='')
            print("         Saving model ...")
        torch.save(model.state_dict(), self.savepath + '/valid_best_checkpoint.pth')
