'''
available arguments:
    --T: context length
    --C: embedding size
    --vocab_size: vocabulary size
    --num_heads: number of attention heads
    --n_layers: number of transformer layers
    --dropout: dropout rate (default: 0.1) 
'''


import torch

from mesoGPT.model import GPT

import argparse

def count_parameters(T, C, vocab_size, num_heads, n_layers, dropout):

    with torch.device("meta"):
        model = GPT(
            T=T,
            C=C,
            vocab_size=vocab_size,
            num_heads=num_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    training_token_budget = total_parameters * 20
    training_compute_budget =  6 * total_parameters * training_token_budget

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Training token budget: {training_token_budget:,}")
    print(f"Training compute budget: {training_compute_budget:,}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Count parameters in a GPT model.")
    parser.add_argument("--T", type=int, required=True, help="Context length")
    parser.add_argument("--C", type=int, required=True, help="Embedding size")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    args = parser.parse_args()

    count_parameters(
        T=args.T,
        C=args.C,
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        n_layers=args.n_layers,
        dropout=args.dropout, 
    )