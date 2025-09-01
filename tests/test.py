from lightning.fabric import Fabric
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TO DO: init to avoid this
from generate import generate
from decoder_model import MultiLayerTransformerDecoder
from tokenizer import Tokenizer

def run_test(weight, outdir, outname):
    
    os.makedirs(outdir, exist_ok=True)

    # List of protein sequences for the 3 tests (harcoded for now)
    high = pd.read_csv('../data/splits/test_90.csv')
    medium = pd.read_csv('../data/splits/test_60_90.csv')
    low = pd.read_csv('../data/splits/test_30_60.csv')
    
    # Prepare the sequences for the test
    high = high.set_index('Uniprot')['Sequence'].to_dict()
    medium = medium.set_index('Uniprot')['Sequence'].to_dict()
    low = low.set_index('Uniprot')['Sequence'].to_dict()
    seqs = {
        'high': high,
        'medium': medium,
        'low': low
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
            training_smiles = pd.read_csv('../data/splits/training_split.csv')['SMILES'].tolist()
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
    