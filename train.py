import torch
import torch.nn as nn
import torch.optim as optim
from decoder_model import MultiLayerTransformerDecoder
from utils.dataset import prepare_data
from utils.earlystopping import EarlyStopping
from utils.configuration import load_config
from utils.timer import Timer
from utils import memory
from tokenizer import Tokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from lightning.fabric import Fabric
from torchinfo import summary
import pandas as pd
import argparse
import wandb
import math
import glob
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, criterion, optimizer, tokenizer, fabric, verbose=1):
    """Train the model for one epoch."""

    model.train()

    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0

    for i, batch in enumerate(dataloader):
        
        if verbose >= 1 and fabric.is_global_zero:
            fabric.print(f"\tT.E.Batch {i+1} of {len(dataloader)} with size {batch['input_ids'].shape[0]}")
            
        input_tensor = batch['input_ids']
        input_att_mask = batch['attention_mask']
        labels = batch['labels']

        logits = model(input_tensor, input_att_mask, tokenizer.delim_token_id, fabric)
        
        batch_size = labels.shape[0]
        seq_length = labels.shape[1]

        # Flatten logits first two dimensions (concatenate seqs from batch)
        logits = logits.contiguous().view(-1, logits.size(-1))
        
        # Flatten masked_labels dimensions (concatenate seqs from batch)
        labels = labels.contiguous().view(-1)
        
        # Compute the loss
        loss = criterion(logits, labels)
        total_train_loss += loss.item()

        # Backward pass and optimization
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True) # set_to_none=True to save memory

        # Accuracy computed only on predicted mol tokens (inlcuding pads)
        delim_idx = (input_tensor[0] == tokenizer.delim_token_id).nonzero().item()
        _, preds = torch.max(logits, dim=1)
        preds = preds.reshape(batch_size, seq_length)
        preds = preds[:, delim_idx:]
        labels = labels.reshape(batch_size, seq_length)
        labels = labels[:, delim_idx:]
        labels[labels==-100] = 1
        
        total_train_correct += (preds == labels).sum().item()
        total_train_samples += labels.numel()

    avg_train_loss = total_train_loss / len(dataloader)
    train_acc = total_train_correct / total_train_samples
    
    return avg_train_loss, train_acc

def scheduled_sampling_prob_invsig(epoch, k):
    """Scheduled sampling probability.
       This is an Inverse Sigmoid Decay Strategy, because it is smoother.
       
       k = decay factor (higher k, faster decay)"""
       
    return k / (k + math.exp(epoch / k))


def scheduled_sampling_prob_lin(epoch):
    """Scheduled sampling probability.
    This is a Linear decay strategy, but adapted to our specific training."""
    
    y = -(0.9/12) * epoch + 0.9
    
    return y

def inverse_scheduled_sampling_prob_lin(epoch, total_epochs):
    """This aims to add a little of teacher forcing as training runs.
    This is a linear increasing strategy, until reaching a total of 0.5 probability
    on the last epoch."""
    
    y = (0.5/total_epochs) * epoch
    
    return y

def train_epoch_scheduled_sampling(model, dataloader, criterion, optimizer, tokenizer, fabric, epoch, num_epochs, scheduled='prob_lin', k=10, verbose=1):
    """
    Train the model for one epoch with teacher forcing, autoregressive, of with scheduled sampling, depending on the flag.
        - scheduled = 0: use teacher forcing always
        - scheduled = 1: do not use teacher forcing
        - scheduled = 'prob_lin': use the linear decay strategy
        - scheduled = 'prob_invsig': use the inverse sigmoid decay strategy
        - scheduled = 'inv_prob_lin': use an increasing linear strategy (inverse scheduled sampling)
    """

    model.train()

    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0

    def run_teacher_forcing(input_tensor, input_att_mask):
        logits = model(input_tensor, input_att_mask, tokenizer.delim_token_id, fabric)
        logits = logits.contiguous().view(-1, logits.size(-1))
        
        return logits
    
    def run_autoregressive(input_tensor, seq_length):
        pad_id = tokenizer.prot_tokenizer.pad_token_id
        delim_idx = (input_tensor[0] == tokenizer.delim_token_id).nonzero().item()

        input_decoder = input_tensor[:, :delim_idx + 1]  # Start with prefix (protein + <del>)
        input_decoder = fabric.to_device(input_decoder)

        predicted_tokens = []

        max_len = seq_length - delim_idx  # Generate only molecule sequence

        for _ in range(max_len):
            input_att_mask = (input_decoder == pad_id)
            input_att_mask = fabric.to_device(input_att_mask)
            logits = model(input_decoder, input_att_mask, tokenizer.delim_token_id, fabric)

            pred_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # Get predicted token
            predicted_tokens.append(pred_token)
            input_decoder = torch.cat((input_decoder, pred_token), dim=1)  # Append prediction

        logits = logits.view(-1, logits.shape[-1])  # Reshape for loss calculation
        
        return logits
    
    if scheduled == 'prob_lin':
        p = scheduled_sampling_prob_lin(epoch)
    elif scheduled == 'prob_invsig':
        p = scheduled_sampling_prob_invsig(epoch,k)
    elif scheduled == 'inv_prob_lin':
        p = inverse_scheduled_sampling_prob_lin(epoch, num_epochs)
    elif scheduled == 0:
        p = 1 # Always use teacher forcing
    elif scheduled == 1:
        p = 0 # Never use teacher forcing (always autoregressive)
    else:
        p = 1 # Default fallback
        fabric.print('WARNING: Incorrect scheduled method, setting to always use teacher forcing.')

    for i, batch in enumerate(dataloader):
        
        if verbose >= 1 and fabric.is_global_zero:
            fabric.print(f"\tT.E.Batch {i+1} of {len(dataloader)} with size {batch['input_ids'].shape[0]}")
            
        input_tensor = batch['input_ids']
        input_att_mask = batch['attention_mask']
        labels = batch['labels']

        batch_size, seq_length = labels.shape
        
        use_teacher_forcing = torch.rand(1).item() < p # Generates a random number in [0, 1)
        
        if verbose >= 2 and fabric.is_global_zero:
            fabric.print(f"\t\tUse teacher forcing: {use_teacher_forcing}, Scheduled Sampling Probability: {p:.4f}")
        
        if use_teacher_forcing:
            logits = run_teacher_forcing(input_tensor, input_att_mask)

        else:
            logits = run_autoregressive(input_tensor, seq_length)

        # Compute loss
        labels = labels.contiguous().view(-1)
        loss = criterion(logits, labels)
        total_train_loss += loss.item()

        # Backward pass and optimization
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # Free memory

        # Accuracy computation (ignoring padding)
        delim_idx = (input_tensor[0] == tokenizer.delim_token_id).nonzero().item()
        _, preds = torch.max(logits, dim=1)
        preds = preds.reshape(batch_size, seq_length)[:, delim_idx:]  # Ignore prefix
        labels = labels.reshape(batch_size, seq_length)[:, delim_idx:]  # Ignore prefix
        labels[labels == -100] = 1  # Ignore loss-padding

        total_train_correct += (preds == labels).sum().item()
        total_train_samples += labels.numel()
        
        fabric.barrier()
        
    avg_train_loss = total_train_loss / len(dataloader)
    train_acc = total_train_correct / total_train_samples
    
    return avg_train_loss, train_acc


def evaluate_epoch(model, dataloader, criterion, tokenizer, fabric, verbose=1):
    """Evaluate the model for one epoch."""

    model.eval()

    total_val_loss = 0

    total_val_predicted = []
    total_val_labels = []

    with torch.no_grad():

        for i, batch in enumerate(dataloader):
            
            if verbose >= 1 and fabric.is_global_zero:
                fabric.print(f"\tV.E.Batch {i+1} of {len(dataloader)} with size {batch['input_ids'].shape[0]}")
                
            input_tensor = batch['input_ids']
            labels = batch['labels']

            pad_id = tokenizer.prot_tokenizer.pad_token_id
            batch_size = labels.shape[0]
            seq_length = labels.shape[1]

            # Start autoregressive eval with prot + <del>
            delim_idx = (input_tensor[0] == tokenizer.delim_token_id).nonzero().item()
            input_decoder = input_tensor[:, :delim_idx+1]
            input_decoder = fabric.to_device(input_decoder)
            predicted = []
            
            max_len = input_tensor.shape[1] - delim_idx
            
            for _ in range(max_len):
                input_att_mask = (input_decoder == pad_id)
                input_att_mask = fabric.to_device(input_att_mask)
                logits = model(input_decoder, input_att_mask, tokenizer.delim_token_id, fabric)
                # Get the predicted token from the last step
                pred_token = logits[:,-1, :].argmax(dim=-1).unsqueeze(1)
                predicted.append(pred_token)
                input_decoder = torch.cat((input_decoder, pred_token), dim=1)
            
            # Get loss
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.contiguous().view(-1)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()
            
            # Get predicted and corresponding labels (only for molecule)
            predicted = torch.cat(predicted, dim=1)
            labels = labels.reshape(batch_size, seq_length)
            labels = labels[:, delim_idx:]
            labels[labels==-100] = 1
            labels = labels.contiguous().view(-1)
            predicted = predicted.contiguous().view(-1)

            total_val_predicted.extend(predicted.cpu().numpy())
            total_val_labels.extend(labels.cpu().numpy())
            correct = (predicted == labels).sum().item()
            
            fabric.barrier()
            
    avg_val_loss = total_val_loss / len(dataloader)
    precision = precision_score(total_val_labels,
                                total_val_predicted,
                                average='macro',
                                zero_division=0)
    recall = recall_score(total_val_labels,
                          total_val_predicted,
                          average='macro',
                          zero_division=0)
    f1 = f1_score(total_val_labels,
                  total_val_predicted,
                  average='macro',
                  zero_division=0)
    accuracy = accuracy_score(total_val_labels, total_val_predicted)

    other_metrics = {'precision': precision, 'recall': recall, 'f1': f1}

    return avg_val_loss, accuracy, other_metrics

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
                label_smoothing=None,
                optimizer='Adam',
                weight_decay=0,
                betas=(0.9, 0.999),
                weights_path='weights/best_model_weights.pth',
                get_wandb=False,
                wandb_project=None,
                wandb_config=None,
                wandb_name='Decoder-only',
                validation_split = 0.2,
                num_gpus=2,
                verbose=False,
                prot_max_length=600,
                mol_max_length=80,
                patience=5,
                delta=0,
                scheduled_sampling=False,
                seed=1234,
                checkpoint_epoch=False,
                resume_training=False
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
        label_smoothing (float, optional): The label smoothing factor for the loss function. Defaults to None.
        optimizer (str, optional): The optimizer to use. Defaults to 'Adam'.
        weight_decay (float, optional): The weight decay (L2 regularization) for optimizer. Defaults to 0.
        betas (tuple, optional): The beta values for the optimizer. Defaults to (0.9, 0.999).
        weights_path (str, optional): The path to save the model weights. Defaults to 'weights/best_model_weights.pth'.
        get_wandb (bool, optional): Whether to log metrics to wandb. Defaults to False.
        wandb_project (str, optional): The wandb project name. Defaults to None.
        wandb_config (dict, optional): The wandb configuration dictionary. Defaults to None.
        wandb_name (str, optional): The wandb run name. Defaults to 'Decoder-only'.
        validation_split (float, optional): The fraction of the data to use for validation. Defaults to 0.2.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 2.
        verbose (int, optional): Different levels of verbosity (0, 1, or 2). Defaults to 0. 
        prot_max_length (int, optional): The maximum length of the protein sequences. Defaults to 600.
        mol_max_length (int, optional): The maximum length of the SMILES strings. Defaults to 80.
        patience (int, optional): The number of epochs to wait before early stopping. Defaults to 5.
        delta (int, optional): The minimum change in validation loss to qualify as an improvement. Defaults to 0.
        scheduled_sampling (bool, optional): Whether to use scheduled sampling. Options: 0, 1, or probability decay function.
        seed (int, optional): The random seed. Defaults to 1234.
        checkpoint_epoch(bool, optional): Whether to save a checkpoint per epoch if True, or overwrite and save the
                                            last epoch checkpoint if False. Defaults to False.
        resume_training (bool, optional): Whether to resume training from a checkpoint. Defaults to False. If true,
                                            the name of the wandb run should be provided.
    """
    fabric = Fabric(accelerator='cuda', devices=num_gpus, num_nodes=1, strategy='ddp', precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(seed, workers=True)

    if get_wandb and fabric.is_global_zero:
        if resume_training:
            wandb_run_id = open(f'wandb/{resume_training}.txt').read().strip()
            wandb.init(
                project=wandb_project,
                config=wandb_config,
                name=wandb_name,
                id=wandb_run_id,
                resume="must"
            )
            fabric.print(f"Resuming wandb run {wandb_run_id}")
        else:
            wandb.init(
                project=wandb_project,
                config=wandb_config,
                name=wandb_name
            )
            with open(f'wandb/{wandb_name}.txt', 'w') as f:
                f.write(wandb.run.id)

    # Tokenizer initialization
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size

    # Data preparation
    train_dataloader, val_dataloader = prepare_data(prot_seqs, smiles,
                                                    validation_split, batch_size,
                                                    tokenizer, fabric, prot_max_length,
                                                    mol_max_length, verbose, seed)

    # Model
    if verbose >=0:
        fabric.print('Initializing the model...')

    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads,
                                         ff_hidden_layer, dropout, num_layers)
    model = fabric.to_device(model)

    assert model.linear.out_features == vocab_size,\
    f"Expected output layer size {vocab_size}, but got {model.linear.out_features}"

    # Print model information
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose >=1 and fabric.is_global_zero:
        summary(model)
        fabric.print(f'The model has {parameters} trainable parameters.')

    # TO DO: Add support for other loss functions and optimizers
    # Loss function
    if loss_function == 'crossentropy':
        if label_smoothing:
            criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        raise ValueError('Invalid loss function. Please use "crossentropy"')

    # Optimizer
    if optimizer == 'Adam':
        if weight_decay:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        if weight_decay and betas:
            optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError('Invalid optimizer. Please use "Adam"')

    # Distribute the model using Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader, use_distributed_sampler=False)
    val_dataloader = fabric.setup_dataloaders(val_dataloader, use_distributed_sampler=False)

    # Check if a checkpoint exists and define starting epoch
    if resume_training and checkpoint_epoch:
        weights_files = glob.glob(weights_path.replace('.pth', '_epoch_*.pth'))
        weights_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
        weights_path = weights_files[-1]
    
    start_epoch = 0
    if resume_training and os.path.exists(weights_path):
        checkpoint = fabric.load(weights_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch

        if verbose >=0:
            fabric.print(f"Resuming training from epoch {start_epoch}")

    # Start the training loop
    if verbose >=0:
        fabric.print('Starting the training loop...')

    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=verbose)

    if fabric.is_global_zero and not resume_training:
        with open(f'metrics/{wandb_name}.csv', 'w') as f:
            f.write('Epoch,T_loss,T_acc,V_loss,V_acc,Pre,Rec,F1\n')
            
    for epoch in range(start_epoch, num_epochs):
        if verbose >=0:
            fabric.print(f"\nEPOCH {epoch+1}/{num_epochs}")

        if verbose >=2 and fabric.is_global_zero:
            timer_epoch = Timer(autoreset=True)
            timer_epoch.start('Epoch %d/%d'%(epoch+1, num_epochs))

        # training
        if verbose >=2 and fabric.is_global_zero:
            timer_train = Timer(autoreset=True)
            timer_train.start('Train Epoch %d/%d'%(epoch+1, num_epochs))

        avg_train_loss, train_acc = train_epoch_scheduled_sampling(model, train_dataloader,
                                                                    criterion, optimizer,
                                                                    tokenizer, fabric,
                                                                    epoch, num_epochs,
                                                                    scheduled_sampling,
                                                                    10, verbose)

        avg_train_acc = fabric.all_reduce(train_acc, reduce_op='mean')

        if verbose >=2 and fabric.is_global_zero:
            timer_train.stop()

        # validation
        if verbose >=2 and fabric.is_global_zero:
            timer_val = Timer(autoreset=True)
            timer_val.start('Validation Epoch %d/%d'%(epoch+1, num_epochs))

        avg_val_loss, val_acc, other_metrics = evaluate_epoch(model, val_dataloader,
                                                              criterion, tokenizer,
                                                              fabric, verbose)
        avg_val_acc = fabric.all_reduce(val_acc, reduce_op='mean')

        if verbose >=2 and fabric.is_global_zero:
            timer_val.stop()

        # Save the metrics into a csv
        if fabric.is_global_zero:
            with open(f'metrics/train_dm{d_model}_nh{num_heads}_ff{ff_hidden_layer}_nl{num_layers}.csv', 'a') as f:
                f.write(
                    f"{epoch + 1},{avg_train_loss},{avg_train_acc},{avg_val_loss},{avg_val_acc},{other_metrics['precision']},{other_metrics['recall']},{other_metrics['f1']}\n"
                )

        # Print the metrics
        if verbose >=1:
            fabric.print(f"Epoch {epoch+1}/{num_epochs}, "\
                        f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}, "\
                        f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}")

            if verbose >=1:
                fabric.print(f"Precision: {other_metrics['precision']:.4f}, "\
                            f"Recall: {other_metrics['recall']:.4f}, F1: {other_metrics['f1']:.4f}")

            if get_wandb and fabric.is_global_zero:
                # log metrics to wandb
                wandb.log({"Epoch": epoch+1, "Train Loss": avg_train_loss,
                        "Train Accuracy": avg_train_acc,
                        "Validation Loss": avg_val_loss,
                        "Validation Accuracy": avg_val_acc,
                        "Validation Precision": other_metrics['precision'],
                        "Validation Recall": other_metrics['recall'],
                        "Validation F1": other_metrics['f1']})

        # Early stopping based on validation loss
        early_stopping(avg_val_loss, model, weights_path, epoch, optimizer, fabric, checkpoint_epoch, resume_training, start_epoch)

        if early_stopping.early_stop:
            if verbose >=0:
                fabric.print(f"Early stopping after {epoch+1} epochs.")
            break

        if verbose >=2 and fabric.is_global_zero:
            timer_epoch.stop()

        if verbose >=2:
            for rank in range(fabric.world_size):
                if fabric.global_rank == rank:
                    memory.get_GPU_memory(device=rank)
                if fabric.is_global_zero:
                    memory.get_CPU_memory()

        fabric.barrier()

    if verbose >=0:
        fabric.print('Training complete!')

def parse_args():
    """Parse the command-line arguments."""

    parser = argparse.ArgumentParser(description='Train a Transformer Decoder model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration YAML file with all the parameters',
                        required=True)
    return parser.parse_args()

def main():
    """Main function to train the model."""
        
    timer_total = Timer(autoreset=True)
    timer_total.start('Total time')

    args = parse_args()

    config = load_config(args.config)

    torch.set_float32_matmul_precision('medium')
    set_seed(config['seed'])
        
    # Get the data
    df = pd.read_csv(config['data_path'])
    prots = df[config['col_prots']].tolist()
    mols = df[config['col_mols']].tolist()

    # Train the model
    train_model(prots, mols, config['num_epochs'], config['learning_rate'],
                config['batch_size'], config['d_model'], config['num_heads'],
                config['ff_hidden_layer'], config['dropout'], config['num_layers'],
                config['loss_function'], config['label_smoothing'],
                config['optimizer'], config['weight_decay'], config['betas'],
                config['weights_path'], config['get_wandb'],config['wandb']['wandb_project'],
                config['wandb']['wandb_config'], config['wandb']['wandb_name'],
                config['validation_split'], config['num_gpus'], config['verbose'],
                config['prot_max_length'], config['mol_max_length'],
                config['es_patience'], config['es_delta'], config['scheduled_sampling'],
                config['seed'], config['checkpoint_epoch'], config['resume_training'])

    timer_total.stop()


if __name__ == '__main__':

    main()
