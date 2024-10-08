import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from decoder_model import MultiLayerTransformerDecoder
from utils.dataset import ProtMolDataset, collate_fn
from utils.earlystopping import EarlyStopping
from utils.configuration import load_config
from tokenizer import Tokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from lightning.fabric import Fabric
from torchinfo import summary
import pandas as pd
import argparse
import time
import wandb


# DATA PREPARATION
def prepare_data(prot_seqs, smiles, validation_split, batch_size, tokenizer,
                 rank, verbose):
    """Prepares datasets, splits them, and returns the dataloaders."""
    
    print('[Rank %d] Preparing the dataset...'%rank)
    dataset = ProtMolDataset(prot_seqs, smiles)

    print('[Rank %d] Splitting the dataset...'%rank)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    if verbose:
        print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, "\
              f"Validation dataset size: {len(val_dataset)}")

    print('[Rank %d] Initializing the dataloaders...'%rank)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=lambda x: collate_fn(x, tokenizer))

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=lambda x: collate_fn(x, tokenizer))

    return train_dataloader, val_dataloader

def train_epoch(model, dataloader, criterion, optimizer, tokenizer, vocab_size,
                teacher_forcing, fabric):
    """Train the model for one epoch."""
    
    model.train()

    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0

    for i, batch in enumerate(dataloader):
        
        input_tensor = batch['input_ids']
        input_att_mask = batch['attention_mask']
        
        # Generate the shifted input tensor for teacher forcing
        # Apply teacher forcing only after the delimiter token
        batch_size = input_tensor.size(0)
        input_tensor_shifted = input_tensor.clone()
        
        if teacher_forcing:
            for i in range(batch_size):
                delim_idx = (input_tensor[i] == tokenizer.delim_token_id).nonzero(as_tuple=True)
                if len(delim_idx[0]) > 0:
                    start_idx = delim_idx[0].item() + 1 # start after the delimiter
                    if start_idx < input_tensor.size(1):
                        # shift the tokens after the delimiter by 1 position
                        input_tensor_shifted[i, start_idx:] = torch.roll(input_tensor[i, start_idx:], shifts=1, dims=0)
                        # the first token after the delimiter should be the padding token
                        input_tensor_shifted[i, start_idx] = tokenizer.mol_tokenizer.pad_token_id
                        # It is not necessary to shift the attention mask
            input_tensor = input_tensor_shifted
        else:
            input_tensor = input_tensor
        
        input_tensor = fabric.to_device(input_tensor)
        input_att_mask = fabric.to_device(input_att_mask)
        
        logits = model(input_tensor, input_att_mask, tokenizer.delim_token_id, fabric)
        
        # calculate the loss just for the second part (after the delimiter)
        # mask after the delimiter
        batch_size = input_tensor.size(0)
        loss_mask = torch.zeros_like(input_tensor, dtype=torch.bool)
        for i in range(batch_size):
            delim_idx = (input_tensor[i] == tokenizer.delim_token_id).nonzero(as_tuple=True)
            if len(delim_idx[0]) > 0:
                start_idx = delim_idx[0].item() + 1
                if start_idx < input_tensor.size(1): # check if there are tokens after the delimiter
                    loss_mask[i, start_idx:] = True
                
        # Apply mask to the logits and labels
        logits = logits.view(batch_size, -1, vocab_size) # [batch_size, seq_len, vocab_size]
        logits = logits[loss_mask] # [num_tokens, vocab_size]
        labels = input_tensor[loss_mask] # [num_tokens]
        
        # Compute the loss
        loss = criterion(logits, labels)
        total_train_loss += loss.item()
        
        # Backward pass and optimization
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True) # set_to_none=True to save memory

        # Calculate the accuracy
        _, preds = torch.max(logits, dim=1)
        total_train_correct += (preds == labels).sum().item()
        total_train_samples += labels.numel()

    avg_train_loss = total_train_loss / len(dataloader)
    train_acc = total_train_correct / total_train_samples

    return avg_train_loss, train_acc

def evaluate_epoch(model, dataloader, criterion, tokenizer, vocab_size, fabric):
    """Evaluate the model for one epoch."""
    
    model.eval()
    
    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_tensor = batch['input_ids']
            input_att_mask = batch['attention_mask']
            
            input_tensor = fabric.to_device(input_tensor)
            input_att_mask = fabric.to_device(input_att_mask)
            
            logits = model(input_tensor, input_att_mask, tokenizer.delim_token_id, fabric)

            # Mask after the delimiter
            batch_size = input_tensor.size(0)
            loss_mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            for i in range(batch_size):
                delim_idx = (input_tensor[i] == tokenizer.delim_token_id).nonzero(as_tuple=True)
                if len(delim_idx[0]) > 0:
                    start_idx = delim_idx[0].item() + 1
                    if start_idx < input_tensor.size(1):
                        loss_mask[i, start_idx:] = True
            
            logits = logits.view(batch_size, -1, vocab_size)
            logits = logits[loss_mask]
            labels = input_tensor[loss_mask]
            
            # Compute the loss
            loss = criterion(logits, labels)

            # Calculate the accuracy
            _, preds = torch.max(logits, dim=1)
            total_val_correct += (preds == labels).sum().item()
            total_val_samples += labels.numel()

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(dataloader)
    val_acc = total_val_correct / total_val_samples
    
    # Calculate additional metrics
    precision = precision_score(labels.cpu(), preds.cpu(),
                                average='macro', zero_division=0)
    recall = recall_score(labels.cpu(), preds.cpu(),
                          average='macro', zero_division=0)
    f1 = f1_score(labels.cpu(), preds.cpu(),
                    average='macro', zero_division=0)   

    other_metrics = {'precision': precision, 'recall': recall, 'f1': f1}

    return avg_val_loss, val_acc, other_metrics

# TRAINING FUNCTION
def train_model(prot_seqs,
                smiles,
                num_epochs=10,
                lr=0.0001,
                batch_size=4,
                d_model=1000,
                num_heads=8,
                ff_hidden_layer=4*1000,
                dropout=0.1,
                num_layers=12,
                loss_function='crossentropy',
                optimizer='Adam',
                weights_path='weights/best_model_weights.pth',
                get_wandb=False,
                teacher_forcing=False,
                validation_split = 0.2,
                num_gpus=2,
                verbose=False
                ):

    """
    Train the model using the specified hyperparameters.

    Args:
        prot_seqs (list): A list of protein sequences
        smiles (list): A list of SMILES strings
        num_epochs (int, optional): The number of epochs to train the model. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.0001.
        batch_size (int, optional): The batch size. Defaults to 4.
        d_model (int, optional): The model dimension. Defaults to 1000.
        num_heads (int, optional): The number of attention heads. Defaults to 8.
        ff_hidden_layer (int, optional): The hidden layer size in the feedforward network. Defaults to 4*d_model.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        num_layers (int, optional): The number of transformer layers. Defaults to 12.
        loss_function (str, optional): The loss function to use. Defaults to 'crossentropy'.
        optimizer (str, optional): The optimizer to use. Defaults to 'Adam'.
        weights_path (str, optional): The path to save the model weights. Defaults to 'weights/best_model_weights.pth'.
        get_wandb (bool, optional): Whether to log metrics to wandb. Defaults to False.
        teacher_forcing (bool, optional): Whether to use teacher forcing. Defaults to False.
        validation_split (float, optional): The fraction of the data to use for validation. Defaults to 0.2.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 2.
        verbose (bool, optional): Whether to print model information. Defaults to False.
    """
    fabric = Fabric(accelerator='cuda', devices=num_gpus, num_nodes=1, strategy='ddp', precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(1234)
    rank = fabric.global_rank

    # Enable memory tracking
    torch.cuda.memory._record_memory_history(max_entries=100000)
    
    # Tokenizer initialization
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Data preparation
    train_dataloader, val_dataloader = prepare_data(prot_seqs, smiles,
                                                    validation_split, batch_size,
                                                    tokenizer, rank, verbose)
    
    # Model
    print('[Rank %d] Initializing the model...'%rank)
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads,
                                         ff_hidden_layer, dropout, num_layers)
    model = fabric.to_device(model)
    
    assert model.linear.out_features == vocab_size,\
    f"Expected output layer size {vocab_size}, but got {model.linear.out_features}"

    # Print model information
    if verbose:
        summary(model)

    # TO DO: Add support for other loss functions and optimizers
    # Loss function
    padding_token_id = tokenizer.mol_tokenizer.token2id['<pad>'] 
    # Ensure that padding tokens are masked during training to prevent the model from learning to generate them.
    if loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss(ignore_index=padding_token_id)
    else:
        raise ValueError('Invalid loss function. Please use "crossentropy"')

    # Optimizer
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError('Invalid optimizer. Please use "Adam"')

    # Distribute the model using Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    # Start the training loop
    print('[Rank %d] Starting the training loop...'%rank)
    best_val_accuracy = 0
    patience = 5
    delta = 0
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=verbose)
    
    for epoch in range(num_epochs):

        # training
        avg_train_loss, train_acc = train_epoch(model, train_dataloader,
                                                criterion, optimizer,
                                                tokenizer, vocab_size,
                                                teacher_forcing, fabric)
    
        # validation
        avg_val_loss, val_acc, other_metrics = evaluate_epoch(model, val_dataloader,
                                                              criterion, tokenizer,
                                                              vocab_size, fabric)

        # Print the metrics
        print(f"[Rank {rank}] Epoch {epoch+1}/{num_epochs}, "\
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "\
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        if verbose:
            print(f"[Rank {rank}] Precision: {other_metrics['precision']:.4f}, "\
                  f"Recall: {other_metrics['recall']:.4f}, F1: {other_metrics['f1']:.4f}")

        if get_wandb:
            # log metrics to wandb
            wandb.log({"Epoch": epoch+1, "Train Loss": avg_train_loss,
                       "Train Accuracy": train_acc,
                       "Validation Loss": avg_val_loss,
                       "Validation Accuracy": val_acc,
                       "Validation Precision": other_metrics['precision'],
                       "Validation Recall": other_metrics['recall'],
                       "Validation F1": other_metrics['f1']})

        # Early stopping based on validation loss
        early_stopping(avg_val_loss, model, weights_path)
        
        if early_stopping.early_stop:
            print(f"[Rank {rank}] Early stopping after {epoch+1} epochs.")
            break

    print('[Rank %d] Training complete!'%rank)
    
    # Disable memory tracking
    if verbose:
        torch.cuda.memory._dump_snapshot('memory_snapshot.pickle')
    torch.cuda.memory._record_memory_history(enabled=None)

def parse_args():
    """Parse the command-line arguments."""
    
    parser = argparse.ArgumentParser(description='Train a Transformer Decoder model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration YAML file with all the parameters',
                        required=True)
    return parser.parse_args()

def main():

    time0 = time.time()

    args = parse_args()

    config = load_config(args.config)

    # Get the data (in this case, it is sampled for testing purposes)
    df = pd.read_csv(config['data_path'])
    df = df.sample(1000)
    prots = df[config['col_prots']].tolist()
    mols = df[config['col_mols']].tolist()

    # Configure wandb if enabled to track the training process
    if config['get_wandb']:
        wandb.init(
            project=config['wandb']['wandb_project'],
            config=config['wandb']['wandb_config']
        )

    # Train the model
    train_model(prots, mols, config['num_epochs'], config['learning_rate'],
                config['batch_size'], config['d_model'], config['num_heads'],
                config['ff_hidden_layer'], config['dropout'], config['num_layers'],
                config['loss_function'], config['optimizer'], config['weights_path'],
                config['get_wandb'], config['teacher_forcing'], config['validation_split'],
                config['num_gpus'], config['verbose'])
    
    timef = time.time() - time0
    print('Time taken:', timef)


if __name__ == '__main__':

    main()
