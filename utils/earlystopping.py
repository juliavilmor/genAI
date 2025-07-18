import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience # Number of epochs to wait before stopping training
        self.delta = delta # Minimum change in the monitored quantity to qualify as an improvement
        self.counter = 0 # Counter to keep track of the number of epochs with no improvement
        self.best_score = np.inf
        self.early_stop = False
        self.verbose = verbose
        
    def __call__(self, score, model, weights_path, epoch, optimizer, fabric, checkpoint_epoch, resume_training, start_epoch=0):

        if self.best_score - score > self.delta:
            if self.verbose >= 1:
                fabric.print(f'EarlyStopping: Validation score improved ({self.best_score:.6f} --> {score:.6f}). '\
                            'Stopping counter to 0.')
            self.best_score = score
            self.counter = 0

        else:
            self.counter += 1
            if self.verbose >= 1:
                fabric.print(f'EarlyStopping: EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        self.save_checkpoint(model, weights_path, epoch, optimizer, fabric, checkpoint_epoch, resume_training, start_epoch)

    def save_checkpoint(self, model, weights_path, epoch, optimizer, fabric, checkpoint_epoch, resume_training, start_epoch):
        state = {
            'model': model,
            'epoch': epoch,
            'optimizer': optimizer,
            }
        if checkpoint_epoch:
            if resume_training:
                weights_path = weights_path.replace(f'{start_epoch}.pth', f'{epoch+1}.pth')
            else:
                weights_path = weights_path.replace('.pth', f'_epoch_{epoch+1}.pth')
            
        fabric.save(weights_path, state)

if __name__ == '__main__':
    early_stopping = EarlyStopping(patience=3, delta=0.01, verbose=True)
    loss = [3.5, 3.4, 3.2, 3.1, 3.3, 3.2, 3.0, 2.6, 2.2, 2.0, 2.0, 2.001, 2.000, 1.9]
    for i, l in enumerate(loss):
        early_stopping(l, '', '')
        if early_stopping.early_stop:
            print(f'early stopping after {i+1} epochs.')
            break
