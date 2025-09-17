import itertools
import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.decoder_model import MultiLayerTransformerDecoder
from utils.tokenizer import Tokenizer


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sweep_parameters(vocab_size, d_model_list, num_heads_list, ff_hidden_list, num_layers_list, dropout=0.1):
    results = []

    # Generate all combinations
    for d_model, num_heads, ff_hidden, num_layers in itertools.product(
        d_model_list, num_heads_list, ff_hidden_list, num_layers_list
    ):
        try:
            # Build model
            model = MultiLayerTransformerDecoder(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                ff_hidden_layer=ff_hidden,
                dropout=dropout,
                num_layers=num_layers,
            )

            # Sanity check
            assert model.linear.out_features == vocab_size, (
                f"Expected output layer size {vocab_size}, but got {model.linear.out_features}"
            )

            # Count parameters
            parameters = count_trainable_parameters(model)

            results.append({
                "d_model": d_model,
                "ff_hidden_layer": ff_hidden,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "ratio": (d_model + ff_hidden * num_layers) / num_layers,
                "parameters": parameters,
            })
        except AssertionError:
            # Skip invalid configurations
            continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by parameters
    df = df.sort_values(by="parameters", ascending=True).reset_index(drop=True)
    return df


# Example usage:
if __name__ == "__main__":
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size

    d_model_list = [128, 256, 512, 640, 768, 1024]
    num_heads_list = [4, 8, 12]
    ff_hidden_list = [256, 512, 1024, 1536, 2048, 2560, 4096, 4608]
    num_layers_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    df = sweep_parameters(vocab_size, d_model_list, num_heads_list, ff_hidden_list, num_layers_list)
    print(df)
    
    # select the rows within a range of parameters
    params = 30000000 # 30M
    error = 2000000   # 2M
    df_selected = df[(df['parameters'] >= params - error) & (df['parameters'] <= params + error)]
    print(df_selected)
