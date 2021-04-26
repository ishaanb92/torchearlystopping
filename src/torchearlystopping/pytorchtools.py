import numpy as np
import torch
from utils.utils import save_model

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_dir=None):
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
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = -1
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, val_loss, model, optimizer=None, scaler=None, scheduler=None, n_iter=-1, n_iter_val=-1, curr_epoch=-1):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = curr_epoch
            save_model(model=model,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       scaler=scaler,
                       n_iter=n_iter,
                       n_iter_val=n_iter_val,
                       epoch=curr_epoch,
                       checkpoint_dir=self.checkpoint_dir,
                       suffix=curr_epoch)

        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else: #Update the best score
            self.best_score = score
            self.best_epoch = curr_epoch
            self.counter = 0
            save_model(model=model,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       scaler=scaler,
                       n_iter=n_iter,
                       n_iter_val=n_iter_val,
                       epoch=curr_epoch,
                       checkpoint_dir=self.checkpoint_dir,
                       suffix=curr_epoch)

        return self.early_stop, self.best_epoch
