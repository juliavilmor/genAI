import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Align import PairwiseAligner

"""
# Get test split > 90% similarity to train data
# Get from the initial clustering to get representative sequences
df = pd.read_csv('../data/data_SMPBind_clean_mapping.csv')
print(df.columns)

# retrieve the proteins that are not in the rep_seq
df = df.drop_duplicates(['Sequence', 'rep_seq'], keep='first')
df = df[df['Sequence'] != df['rep_seq']]
print(df)

# From these 3048 sequences, select 100 different between them
df = df.sample(n=100)
print(df)
print(df.columns)
df = df[['Uniprot', 'Compound_ID', 'SMILES', 'Sequence']]
df.to_csv('../data/splits/test_90.csv', index=False)
"""

# Extract from the training dataset the 3 other test splits
# based of the sequence id (60-90, 30-60, and <30)

def get_sequences_from_range_perc_id(dataset, output_file, range): 
    """
    In range, select one of these 3 options: 60-90, 30-60, or 0-30.
    """
    df = pd.read_csv(dataset)

    # get just the unique protein sequences
    df = df.drop_duplicates(['Sequence'], keep='first')

    # Define and compute the percentage identity
    aligner = PairwiseAligner()
    aligner.mode = 'global'

    def calculate_identity(seq1, seq2):
        alignment = aligner.align(seq1, seq2)[0]
        matches = sum(1 for a, b in zip(alignment[0], alignment[1]) if a == b)
        return matches / max(len(seq1), len(seq2)) * 100

    try:
        with open(output_file, 'r') as f:
            stored_sequences = f.read().splitlines()
    except FileNotFoundError:
        stored_sequences = []

    # Fill the file until it has 100 sequences
    while len(stored_sequences) < 100:
        
        # get one random sequence and calculate the percentage id
        random_seq = df['Sequence'].sample(n=1).iloc[0]
        if random_seq in stored_sequences: continue # skip if already in the file
        df = df[df['Sequence'] != random_seq] # extract random sequence from the dataframe

        df['perc_id'] = df['Sequence'].apply(lambda seq: calculate_identity(random_seq, seq))
        mean_perc_id = df['perc_id'].max()
        
        print(mean_perc_id, random_seq)

        # get range
        top = int(range.split('-')[-1])
        bott = int(range.split('-')[0])
        
        # store the sequence if it is inside the percentage range
        if bott <= mean_perc_id < top:
            stored_sequences.append(random_seq)
            with open(output_file, 'a') as f:
                f.write(f'{random_seq}\n')
                print(f'Added sequence {len(stored_sequences)}')
    

def extract_sequences_from_dataset(dataset, sequences, output):
    
    df = pd.read_csv(dataset)
    print(len(df))
    
    with open(sequences, 'r') as f:
        seqs = f.read().splitlines()
    
    df_filt = df[~df['Sequence'].isin(seqs)]
    df_filt.to_csv(output, index=False)
    print(len(df_filt))
    
def map_sequences_to_ids(sequences_file, dataset_file, output_file):
    df = pd.read_csv(dataset_file)
    print(len(df))
    
    with open(sequences_file, 'r') as f:
        seqs = f.read().splitlines()
    
    df_filt = df[df['Sequence'].isin(seqs)]
    df_filt = df_filt.drop_duplicates(['Sequence'], keep='first')
    print(len(df_filt))
    print(df_filt)
    df_filt.to_csv(output_file, index=False)


if __name__ == '__main__':
    #get_sequences_from_range_perc_id('../data/data_SMPBind_downsampled_50.csv', '../data/splits/sequences_60_90.txt', '60-90')
    #get_sequences_from_range_perc_id('../data/data_SMPBind_downsampled_50.csv', '../data/splits/sequences_30_60.txt', '30-60')
    #get_sequences_from_range_perc_id('../data/data_SMPBind_downsampled_50.csv', '../data/splits/sequences_0_30.txt', '0-30')
    
    #extract_sequences_from_dataset('../data/data_SMPBind_downsampled_50.csv', '../data/splits/sequences_60_90.txt', '../data/splits/tmp.csv')
    #extract_sequences_from_dataset('../data/splits/tmp.csv', '../data/splits/sequences_30_60.txt', '../data/splits/training_split.csv')
    
    map_sequences_to_ids('../data/splits/sequences_60_90.txt', '../data/data_SMPBind_downsampled_50.csv', '../data/splits/test_60_90.csv')
    map_sequences_to_ids('../data/splits/sequences_30_60.txt', '../data/data_SMPBind_downsampled_50.csv', '../data/splits/test_30_60.csv')