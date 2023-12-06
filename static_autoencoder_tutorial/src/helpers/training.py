import numpy as np 
import torch 
import torch.nn as nn
from copy import deepcopy
from loguru import logger
import os 
from pathlib import Path

class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, epochs, early_stopping=None):
        self.model = model 
        self.optimizer=optimizer
        self.loss_fn=loss_fn
        self.train_loader = train_loader 
        self.val_loader = val_loader
        self.num_epochs = epochs

        self.train_losses=[]
        self.val_losses=[]
        self.avg_val_losses=[]
        self.curr_epoch = 0
        self.train_size = len(train_loader)
        self.val_size=len(val_loader)
        self.early_stopping = early_stopping
        self.best_model = deepcopy(model)

    def train(self):
        for t in range(self.num_epochs):
            self.current_epoch = t
            continue_running_loop = self._run_single_loop()
            if not continue_running_loop:
                print('Stop running loop')
                break 
        
        # Define the top performing model
        # If early stopping is set, we load the model 
        #   from the path specified in self.early_stopping
        if self.early_stopping is not None:
            logger.info(f'Early stopping is set: loading best model from {self.early_stopping.path}')
            self.best_model.load_state_dict(torch.load(self.early_stopping.path))
        else:
            self.best_model = self.model 
    
    def _run_single_loop(self):
        # Set the model to training mode - important for batch normalization and dropout layers
        self.model.train()
        for batch, (X, y) in enumerate(self.train_loader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, X)
            self.train_losses.append(loss.item())

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1)
                print(f"train loss: {loss:>7f}  [{current:>5d}/{self.train_size:>5d}]")
        
        # Do validation step
        self.model.eval()
        with torch.no_grad():
            batch_val_loss=[]
            for batch, (X, y) in enumerate(self.val_loader):
                # Compute prediction and loss
                pred = self.model(X)
                loss = self.loss_fn(pred, X)
                self.val_losses.append(loss.item())
                batch_val_loss.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        if self.early_stopping:
            logger.info(f"Size of batch validation: {len(batch_val_loss)}")
            logger.info(f"First 5 rows of batch validation loss: {batch_val_loss[:5]}")
            self.early_stopping(np.average(batch_val_loss), self.model)
        
            if self.early_stopping.early_stop:
                print("Early stopping")
                return False
        
        return True 
    
    def plot_learning_curves(
        self,
        loss_type='train'
    ):
        fig, ax = plt.subplots(1)
        if loss_type=='train':
            y=self.train_losses
        else:
            y=self.val_losses
        x=list(range(len(y)))
        ax.lineplot(
            x=x,
            y=y 
        )
        ax.set_title(loss_type)
        plt.show()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=logger.info):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        # Create path for saving model 
        if self.path is not None:
            Path(self.path).mkdir(parents=True, exist_ok=True)
            self.path = Path(self.path, 'checkpoint.pt')
        else:
            self.path = 'checkpoint.pt'

    def __call__(self, val_loss, model):

        score = -val_loss

        
        if self.best_score is None:
            self.trace_func(f'best_score is None, saving best score...')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.trace_func(f'current score {score} is lower than {self.best_score + self.delta}, incrementing patience counter')
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.trace_func(f'current score {score} is higher than {self.best_score + self.delta}, setting new best score to {score}')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss