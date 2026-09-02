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

from experiments.exp_002.model import GPT, GPTConfig

import argparse
import math

def count_parameters(B, T, C, vocab_size, num_heads, n_layers, dropout):

    GPT_config = GPTConfig(
        T=T,
        C=C,
        vocab_size=vocab_size,
        num_heads=num_heads,   
        n_layers=n_layers,
        dropout=dropout,
    )

    with torch.device("meta"):
        model = GPT(
            GPT_config
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

    model_state_memory = 16 * total_parameters
    run_time_memory = n_layers * (B * T * C * (66 + ((9 * num_heads * T) / C))) + (8 * T * B * C) + (4 * T * B * vocab_size)
    run_time_memory_ = n_layers * (B * T * C * (34 + ((5 * num_heads * T) / C))) + (4 * T * B * C) + (2 * T * B * vocab_size)
    temporary_memory = 20 * 1_073_741_824
    total_memory_usage = model_state_memory + run_time_memory + temporary_memory
    total_memory_usage_ = model_state_memory + run_time_memory_ + temporary_memory

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Training token budget: {training_token_budget:,}")
    print(f"Training compute budget: {training_compute_budget:,}")
    print(f"For batch size {B}, Optimizer steps needed: {math.ceil(training_token_budget / (B * T)):,}")
    print(f"Total memory usage (in GiB) - FP-32: {(total_memory_usage / 1_073_741_824):,}")
    print(f"Total memory usage (in GiB) - FP-16: {(total_memory_usage_ / 1_073_741_824):,}")
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Count parameters in a GPT model.")
    parser.add_argument("--B", type=int, required=False, default=1, help="Batch size")
    parser.add_argument("--T", type=int, required=True, help="Context length")
    parser.add_argument("--C", type=int, required=True, help="Embedding size")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, required=False, default=0.1, help="Dropout rate")

    args = parser.parse_args()

    count_parameters(
        B=args.B,
        T=args.T,
        C=args.C,
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        n_layers=args.n_layers,
        dropout=args.dropout, 
    )