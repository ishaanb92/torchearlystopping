import numpy as np
import torch

def save_model(model=None,
               optimizer=None,
               scheduler=None,
               scaler=None,
               n_iter=None,
               n_iter_val=None,
               epoch=None,
               checkpoint_dir=None,
               suffix=None):
    """
    Function save the PyTorch model along with optimizer state

    Parameters:
        model (torch.nn.Module object) : Pytorch model whose parameters are to be saved
        optimizer (torch.optim object) : Optimizer used to train the model
        epoch (int) : Epoch the model was saved in
        path (str or Path object) : Path to directory where model parameters will be saved
        suffix (str): Suffix string to prevent new save from overwriting old ones

    Returns:
        None

    """

    if model is None:
        print('Save operation failed because model object is undefined')
        return

    if optimizer is None:
        print('Save operation failed because optimizer is undefined')
        return

    if scaler is None:
        scaler_state_dict = None
    else:
        scaler_state_dict = scaler.state_dict()

    if scheduler is None:
        save_dict = {'n_iter': n_iter,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': None,
                     'scaler': scaler_state_dict,
                     'n_iter_val': n_iter_val,
                     'epoch': epoch
                    }
    else:
        save_dict = {'n_iter': n_iter,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(),
                     'scaler': scaler_state_dict,
                     'n_iter_val': n_iter_val,
                     'epoch': epoch
                     }

    #  Overwrite existing checkpoint file to avoid running out of memory
    if suffix is None:
        fname = 'checkpoint.pt'
    else:
        fname = 'checkpoint_{}.pt'.format(suffix)


    save_path = os.path.join(checkpoint_dir, fname)

    torch.save(save_dict, save_path)



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
                       suffix=None)

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
                       suffix=None)

        if self.verbose is True:
            print('Current:: Score = {} Epoch = {}'.format(score, curr_epoch))
            print('Best:: Score = {} Epoch = {}'.format(self.best_score, self.best_epoch))

        return self.early_stop, self.best_epoch
