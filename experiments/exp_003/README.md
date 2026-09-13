# Experiment 003:  Accelerating Attention with PyTorch SDPA

## Summary

This experiment evaluated PyTorch’s scaled dot-product attention (SDPA) as a replacement for the explicit causal-attention implementation used in [Baseline Language-Model Pretraining](../exp_002/README.md). The model architecture and training configuration were otherwise unchanged.

On a single NVIDIA L40S GPU, the best-performing SDPA configuration used a micro-batch size of 16. Compared with the baseline, SDPA reduced the optimizer-update time from 17.16 to 4.20 seconds, increased training throughput from 26,093 to 108,781 tokens/s, and raised the estimated training-only model FLOP utilization (MFU) from 4.94% to 20.20%. Peak GPU memory usage decreased from 27,969 to 12,457 MiB.

This corresponds to approximately:

- 4.17× higher training throughput;
- 4.09× faster optimizer-update time;
- 2.25× lower peak GPU memory.

Micro-batch sizes 32 and 64 were also tested. Neither exceeded the performance
of micro-batch size 16.

These were short performance runs rather than complete pretraining runs.
Consequently, this experiment evaluates training-system efficiency not final model quality or convergence.

## Runs Compared

| Role | Experiment | Run ID | Attention | Micro-batch | Accumulation | Recorded steps |
|---|---|---|---|---:|---:|---:|
| Full-run baseline | `exp_002` | `2026-09-05_11-41-45_UTC` | Explicit attention | 16 | 32 | 3,726 |
| SDPA benchmark | `exp_003` | `2026-09-13_13-24-25_UTC` | PyTorch SDPA | 16 | 32 | 37 |
| SDPA benchmark | `exp_003` | `2026-09-13_13-30-42_UTC` | PyTorch SDPA | 32 | 16 | 35 |
| SDPA benchmark | `exp_003` | `2026-09-13_13-35-16_UTC` | PyTorch SDPA | 64 | 8 | 35 |


## Experimental Design

The model architecture mentioned in [Baseline Language-Model Pretraining](../exp_002/README.md) was used in this experiment.

| Variable | Value |
|---|---:|
| Trainable parameters | 97,655,296 |
| Transformer blocks | 12 |
| Model width | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Context length | 1,024 |
| Vocabulary size | 16,384 |
| Global batch size | 512 sequences |
| Tokens per optimizer update | 524,288 |
| Precision | BF16 mixed precision |
| Dropout | 0.1 |

One variables changed:

- Explicit causal attention was replaced with PyTorch SDPA.

The performance with SDPA was evaluated across different micro-batch sizes 16, 32, and 64 while preserving a global batch size of 512.


## Change Introduced

The baseline explicitly implemented the attention mechanism:

    QKᵀ → scale → causal mask → softmax → dropout → multiply by V

`exp_003` replaced this sequence with:

    torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=...,
        is_causal=True,
    )

No learned layers or model dimensions were changed. The causal-mask buffer was
removed because causal masking is handled internally by SDPA.


## Performance Results

| Attention | Micro-batch | max update time (excluding 1st step) | Throughput | Estimated MFU | Peak GPU memory | Estimated Training Time | Source |
|---|---:|---:|---:|---:|---:|---:|---:|
| Explicit attention | 16 | 17.16 s | 26,093 tok/s | 4.94% | 27,969 MiB | 20.797 hrs | exp_002 - Run ID: `2026-09-05_10-22-52_UTC`
| SDPA | 16 | **4.20 s** | **108,781 tok/s** | **20.20%** | **12,457 MiB** | 4.988 hrs | exp_003 - Run ID: `2026-09-13_13-24-25_UTC`
| SDPA | 32 | 4.71 s | 86,782 tok/s | 18.01% | 21,715 MiB | 6.253 hrs | exp_003 - Run ID: `2026-09-13_13-30-42_UTC`
| SDPA | 64 | 5.11 s | 67,071 tok/s | 16.60% | 42,369 MiB | 8.090 hrs | exp_003 - Run ID: `2026-09-13_13-35-16_UTC`

### Improvement Relative to the Baseline

| SDPA micro-batch | Training Throughput | Optimizer Update-time | Peak GPU memory |
|---:|---:|---:|---:|
| 16 | **4.17×** | **4.09×** | **2.25× lower** |
| 32 | 3.33× | 3.64× | 1.29× lower |
| 64 | 2.57× | 3.36× | 1.51× higher |


Micro-batch size 16 was the best configuration tested. It simultaneously
provided the highest throughput, highest MFU, and lowest peak memory
usage. Increasing the micro-batch size did not improve Model FLOP Utilization.