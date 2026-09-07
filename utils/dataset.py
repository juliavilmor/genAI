import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split, DistributedSampler
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DATASET CLASS
class ProtMolDataset(Dataset):
    def __init__(self, prot_seqs, smiles):
        self.prot_seqs = prot_seqs
        self.smiles = smiles
    def __len__(self):
        return len(self.prot_seqs)
    def __getitem__(self, idx):
        prot_seq = self.prot_seqs[idx]
        smile = self.smiles[idx]
        return prot_seq, smile

# COLLATE FUNCTION WITH TOKENIZER AND TEACHER FORCING IMPLEMENTED
def filter_sequences_by_unknown_tokens(input_ids, attention_mask, unknown_token_id, pad_token_id, threshold=0.2):
    
    unknown_token_count = (input_ids == unknown_token_id).sum(dim=1)
    total_token_count = (input_ids != pad_token_id).sum(dim=1)
    # the threshold is the ratio of unknown tokens to total tokens
    unknown_token_ratio = unknown_token_count / total_token_count
    valid_sequences_mask = unknown_token_ratio < threshold
    
    filtered_input_ids = input_ids[valid_sequences_mask]
    filtered_attention_mask = attention_mask[valid_sequences_mask]
    
    return {'input_ids': filtered_input_ids, 'attention_mask': filtered_attention_mask}

def collate_fn(batch, tokenizer, prot_max_length, mol_max_length):
    # Tokenize the protein sequences and SMILES strings
    prot_seqs = [prot_seq for prot_seq, _ in batch]
    smiles = [smile for _, smile in batch]

    encoded_texts = tokenizer(prot_seqs, smiles,
                              prot_max_length=prot_max_length,
                              mol_max_length=mol_max_length)

    encoded_texts = filter_sequences_by_unknown_tokens(encoded_texts['input_ids'],
                                                       encoded_texts['attention_mask'],
                                                       tokenizer.prot_tokenizer.unk_token_id,
                                                       tokenizer.prot_tokenizer.pad_token_id,
                                                       threshold=0.2)
    
    # teacher forcing by removing last element of input_ids and first element of labels
    input_ids = encoded_texts['input_ids'][:, :-1]
    attention_mask = encoded_texts['attention_mask'][:, :-1]
    labels = encoded_texts['input_ids'][:, 1:]

    # get labels with -100 (ignore_index from loss) to all protein tokenids
    protein_ids = tokenizer.prot_ids + [tokenizer.prot_tokenizer.pad_token_id] + [tokenizer.delim_token_id]
    labels = torch.where(torch.isin(labels, torch.tensor(protein_ids)), -100, labels)
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def sort_by_molecules_per_protein(dataset, verbose=0, out='splits/train_split.csv'):
    """Group the dataset by protein, ordering proteins by how many distinct
    molecules they have. Returns a Subset in that order."""

    # The DataFrame index is the dataset index, so it survives the sort and can
    # be handed straight to Subset.
    df = pd.DataFrame({'sequence': [dataset[idx][0] for idx in range(len(dataset))],
                       'smiles': [dataset[idx][1] for idx in range(len(dataset))]})
    df['num_molecules'] = df.groupby('sequence')['smiles'].transform('nunique')
    sorted_df = df.sort_values(by=['num_molecules', 'sequence'],
                               ascending=[True, True], kind='stable')

    if verbose >= 2:
        # Print the sorted DataFrame for debugging
        print("Sorted DataFrame by number of molecules per protein:")
        print(sorted_df.head())

        # Save the splits into a csv
        sorted_df.drop(columns=['num_molecules']).to_csv(out, index=False)

    return Subset(dataset, sorted_df.index.tolist())

# DATA PREPARATION
def prepare_data(prot_seqs, smiles, validation_split, batch_size, tokenizer,
                 fabric, prot_max_length, mol_max_length, verbose, seed):
    """Prepares datasets, splits them, and returns the dataloaders."""
    
    if verbose >=0:
        fabric.print('Preparing the dataset...')
    dataset = ProtMolDataset(prot_seqs, smiles)

    if verbose >=0:
        fabric.print('Splitting the dataset...')
    
    # Split the dataset into train and validation randomly
    torch.manual_seed(seed)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Sort the datasets according to the number of molecules per protein
    if verbose >= 0:
        fabric.print('Sorting the datasets by the number of molecules per protein...')
    
    if len(val_dataset) > 0: # in the case there is no validation
        val_dataset = sort_by_molecules_per_protein(val_dataset, verbose, out='splits/val_split.csv')
        idxs_val = list(range(len(val_dataset)))
        
    train_dataset = sort_by_molecules_per_protein(train_dataset, verbose, out='splits/train_split.csv')
    idxs_train = list(range(len(train_dataset)))
    
    # Check if the sort worked
    if verbose >= 2:
        fabric.print('Lenghts of the sequences in the train and validation datasets:')
        train_lengths = [len(train_dataset[idx][0]) + len(train_dataset[idx][1]) for idx in idxs_train]
        fabric.print(np.asarray(train_lengths))
        if len(val_dataset) > 0:
            val_lengths = [len(val_dataset[idx][0]) + len(val_dataset[idx][1]) for idx in idxs_val]
            fabric.print(np.asarray(val_lengths))
    
    # Plot the distribution of the lengths of the proteins and molecules in the train and validation datasets
    if verbose >= 2:
        fabric.print('Plotting the distribution of the lengths of the proteins and molecules...')
        plt.figure()    
        train_prots_lenghts = [len(train_dataset[idx][0]) for idx in idxs_train]
        plt.hist(train_prots_lenghts, bins=75, label='train proteins', alpha=0.5)
        if len(val_dataset) > 0:
            val_prots_lenghts = [len(val_dataset[idx][0]) for idx in idxs_val]
            plt.hist(val_prots_lenghts, bins=75, label='val proteins', alpha=0.5)
        plt.legend()
        plt.savefig('plots/hist_train_test_prots.pdf')
        
        plt.figure()
        train_mols_lenghts = [len(train_dataset[idx][1]) for idx in idxs_train]
        plt.hist(train_mols_lenghts, bins=100, label='train mols',alpha=0.5)
        if len(val_dataset) > 0:
            val_mols_lenghts = [len(val_dataset[idx][1]) for idx in idxs_val]
            plt.hist(val_mols_lenghts, bins=100, label='val mols', alpha=0.5)
        plt.legend()
        plt.savefig('plots/hist_train_test_mols.pdf')
    
    if verbose >=1:
        fabric.print(f"Train dataset size: {len(train_dataset)}, "\
                    f"Validation dataset size: {len(val_dataset)}")
    
    if verbose >=0:
        fabric.print('Initializing the dataloaders...')
    
    sampler_train = DistributedSampler(train_dataset, shuffle=False, drop_last=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  sampler=sampler_train,
                                  collate_fn=lambda x: collate_fn(x, tokenizer,
                                                                  prot_max_length,
                                                                  mol_max_length))
    
    sampler_val = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=sampler_val,
                                collate_fn=lambda x: collate_fn(x, tokenizer,
                                                                prot_max_length,
                                                                mol_max_length))

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    # Self-check: the sort must actually group rows by protein, ordering proteins
    # by their number of distinct molecules (ascending).
    prot_seqs = ['B', 'A', 'C', 'A', 'B', 'A']
    smiles = ['b1', 'a1', 'c1', 'a2', 'b2', 'a3']
    sorted_ds = sort_by_molecules_per_protein(ProtMolDataset(prot_seqs, smiles))
    got = [sorted_ds[i] for i in range(len(sorted_ds))]

    assert [p for p, _ in got] == ['C', 'B', 'B', 'A', 'A', 'A'], got
    assert sorted(got) == sorted(zip(prot_seqs, smiles)), 'sort must be a permutation'
    print('ok')
