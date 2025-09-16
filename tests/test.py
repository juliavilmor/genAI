from lightning.fabric import Fabric
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate import generate
from models.decoder_model import MultiLayerTransformerDecoder
from utils.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def run_test(weight, outdir, outname):
    
    os.makedirs(outdir, exist_ok=True)

    # List of protein sequences for the 3 tests (harcoded for now)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    high = pd.read_csv(os.path.join(current_dir, '../data/splits/test_90.csv'))
    medium = pd.read_csv(os.path.join(current_dir, '../data/splits/test_60_90.csv'))
    low = pd.read_csv(os.path.join(current_dir, '../data/splits/test_40_60.csv'))
    very_low = pd.read_csv(os.path.join(current_dir, '../data/splits/test_0_40.csv'))
    
    # Prepare the sequences for the test
    high = high.set_index('Uniprot')['Sequence'].to_dict()
    medium = medium.set_index('Uniprot')['Sequence'].to_dict()
    low = low.set_index('Uniprot')['Sequence'].to_dict()
    very_low = very_low.set_index('Uniprot')['Sequence'].to_dict()
    seqs = {
        'high': high,
        'medium': medium,
        'low': low,
        'very_low': very_low
    }

    # Define total molecules to generate (harcoded for now)
    Ngen = 100

    # Define the hyperparameters --> take them from the weights name
    d_model        = int(weight.split('_')[1].split('dm')[1])
    num_heads      = int(weight.split('_')[2].split('nh')[1])
    ff_hidden_layer  = int(weight.split('_')[3].split('ff')[1])
    num_layers     = int(weight.split('_')[4].split('nl')[1].split('.')[0])
    dropout       = 0.25
    print(d_model, num_heads, ff_hidden_layer, num_layers)

    # Initialize the model
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size

    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers)

    # Load the model weights
    fabric = Fabric(accelerator='cuda', devices=[0])
    fabric.launch()
    state = {'model': model}
    fabric.load(weight, state)

    # Generate SMILES strings for each protein sequence
    results = []
    for test_type, data in seqs.items():
        print(f'Generating molecules for {test_type} identity sequences...')
        
        for i, seq in data.items():
            print(test_type, i)
            generated_smiles, sucess_rate = generate(Ngen, seq, model, fabric, max_length=80,\
                                                    temperature=1.2, verbose=False,\
                                                    outdir=outdir,\
                                                    outname='generated_molecules_%s_seq_%s.csv'%(test_type, i))
            
            # Add Ngen, Nval, Nuniq, Nunk
            Nval = len(generated_smiles)
            Nuniq = len(set(generated_smiles['smiles'].tolist())) # PENDING: filter by molecular analysis
            training_smiles = pd.read_csv(os.path.join(current_dir, '../data/splits/training_split.csv'))['SMILES'].tolist()
            Nunk = len(set(generated_smiles['smiles'].tolist()) - set(training_smiles))
            print('Generated, Valid, Unique, Novel:')
            print(Ngen, Nval, Nuniq, Nunk)
            
            # Add Val, Uniq, Novel
            validity = Nval / Ngen * 100 if Ngen > 0 else 0
            uniqueness = Nuniq / Nval * 100 if Nval > 0 else 0
            novelty = Nunk / Nuniq * 100 if Nuniq > 0 else 0
            print('Validity, Uniqueness, Novelty (%):')
            print(validity, uniqueness, novelty)
            
            # Save results
            results.append([test_type, i, seq, Ngen, Nval, Nuniq, Nunk, validity, uniqueness, novelty])
            
    df_results = pd.DataFrame(results, columns=['test_type', 'seq_idx', 'sequence', 'Ngen', 'Nval', 'Nuniq', 'Nunk', 'validity', 'uniqueness', 'novelty'])
    df_results.to_csv(f'{outdir}/{outname}.csv', index=False)
    
def analyse_test(weight, results_csv, outdir):
    # Load the results
    df = pd.read_csv(results_csv)
    
    # Compute the mean, median, mode, and standard deviation of the validity, uniqueness, and novelty
    stats = {}
    for metric in ['validity', 'uniqueness', 'novelty']:
        stats[metric] = {
            'mean': df[metric].mean(),
            'median': df[metric].median(),
            'mode': df[metric].mode()[0],
            'std': df[metric].std()
        }
    
    print(f'Statistics (general) for model weights: {weight}')
    for metric, values in stats.items():
        print(f'{metric.capitalize()}:')
        for stat, value in values.items():
            print(f'  {stat}: {value}')
    
    # Compute the metrics for each test type
    for test_type in df['test_type'].unique():
        df_test = df[df['test_type'] == test_type]
        print(f'\nStatistics for test type: {test_type}')
        for metric in ['validity', 'uniqueness', 'novelty']:
            mean = df_test[metric].mean()
            median = df_test[metric].median()
            mode = df_test[metric].mode()[0]
            std = df_test[metric].std()
            print(f'{metric.capitalize()}:')
            print(f'  Mean: {mean}')
            print(f'  Median: {median}')
            print(f'  Mode: {mode}')
            print(f'  Std: {std}')
    
    # Save the statistics as a report
    with open(f'{outdir}/statistics_{weight.split("/")[-1].split(".")[0]}.txt', 'w') as f:
        f.write(f'Statistics (general) for model weights: {weight}\n')
        for metric, values in stats.items():
            f.write(f'{metric.capitalize()}:\n')
            for stat, value in values.items():
                f.write(f'  {stat}: {value}\n')
        f.write('\n')
        for test_type in df['test_type'].unique():
            df_test = df[df['test_type'] == test_type]
            f.write(f'Statistics for test type: {test_type}\n')
            for metric in ['validity', 'uniqueness', 'novelty']:
                mean = df_test[metric].mean()
                median = df_test[metric].median()
                mode = df_test[metric].mode()[0]
                std = df_test[metric].std()
                f.write(f'{metric.capitalize()}:\n')
                f.write(f'  Mean: {mean}\n')
                f.write(f'  Median: {median}\n')
                f.write(f'  Mode: {mode}\n')
                f.write(f'  Std: {std}\n')
            f.write('\n')
                
    # Plot a boxplot for each test type and each metric
    for metric in ['validity', 'uniqueness', 'novelty']:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='test_type', y=metric, data=df)
        plt.title(f'Boxplot of {metric.capitalize()} by Test Type')
        plt.ylabel(f'{metric.capitalize()} (%)')
        plt.xlabel('Test Type')
        plt.ylim(0, 100)
        plt.savefig(f'{outdir}/boxplot_{metric}_{weight.split("/")[-1].split(".")[0]}.png')
        plt.close()
        
    # For each test type, plot the metric per sequence index
    for test_type in df['test_type'].unique():
        df_test = df[df['test_type'] == test_type]
        for metric in ['validity', 'uniqueness', 'novelty']:
            plt.figure(figsize=(20, 5))
            sns.barplot(x='seq_idx', y=metric, data=df_test)
            plt.title(f'{metric.capitalize()} per Sequence Index for Test Type: {test_type}')
            plt.ylabel(f'{metric.capitalize()} (%)')
            plt.xlabel('Sequence Index')
            plt.ylim(0, 100)
            plt.xticks(rotation=90)
            plt.savefig(f'{outdir}/seqs_{metric}_{test_type}_{weight.split("/")[-1].split(".")[0]}.png')
            plt.close()

if __name__ == "__main__":
    
    # Define arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run generation test.")
    parser.add_argument('--weights_file', type=str, required=True, help='Path to the model weights file.')
    parser.add_argument('--outdir', type=str, required=True, default='test_generation', help='Directory to save the results.')
    parser.add_argument('--outname', type=str, required=True, default='results_test', help='Name of the results file.')
    args = parser.parse_args()
    
    weights_file = args.weights_file
    outdir = args.outdir
    outname = args.outname
    
    # Run the test function
    run_test(weights_file, outdir, outname)
    analyse_test(weights_file, f'{outdir}/{outname}.csv', outdir)
    
