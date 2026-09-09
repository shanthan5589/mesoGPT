# Experiment: Baseline Language-Model Pretraining

## Summary

This experiment trained a **97.7M-parameter decoder-only Transformer** from scratch on approximately **1.95B tokens**. The purpose of the run was to establish a reproducible baseline for subsequent architecture and data pipeline changes.

Training completed successfully on a single NVIDIA L40S in **20h 47m 48s**. The best checkpoint reached a validation loss of **3.5299** and validation BPB of **1.1509** at step 3,720.

![Losses](./assets/losses.png)
*Figure 1. **Left**:  Training and validation loss throughout pretraining. **Right**: (Train loss - Validation loss) throughout pretraining.*

*Note: The train and validation curves almost perfectly overlap because validation shard was drawn from the same distribution as the training shards. It is not because of evaluating the same tensors twice (see graph on the right).*

| Result | Value |
|---|---:|
| Commit | `18f0519` |
| Trainable parameters | 97,655,296 |
| Optimizer steps | 3,726 |
| Estimated tokens processed | 1,953,497,088 |
| Tokens per parameter ratio | 20.004 |
| Best validation loss | **3.5299** |
| Best validation BPB | **1.1509** |
| Wall-clock training time | 20h 47m 48s |
| Average end-to-end throughput | ~26,093 tokens/s |
| Training compute budget | ~1.145 EFLOP |
| Achieved model throughput | ~15.3 TFLOP/s |
| Achieved model FLOP utilization | 4% (~15.3 TFLOP/s) |
| Training cost | $39.08 |

## Objective

Establish a reproducible baseline for subsequent architecture and data pipeline changes.

## Hypothesis

A 12-layer, 768-dimensional decoder-only Transformer trained on approximately 20 tokens per parameter should converge smoothly and provide a stable reference point for evaluating future changes to the architecture, optimizer, data pipeline, and training-system efficiency.

## Run Identity and Reproducibility

| Field | Value |
|---|---|
| Experiment ID | `exp_002` |
| Run ID | `2026-09-05_11-41-45_UTC` |
| Start time | 2026-09-05 11:41:45 UTC |
| End time | 2026-09-06 08:29:36 UTC |
| Training precision | BF16 Mixed Precision |


## Data

| Field | Value |
|---|---|
| Dataset | `karpathy/climbmix-400b-shuffle` |
| License | MIT  |
| Documents per shard | ~86,016 |
| Raw characters per shard | ~252.6M |
| Tokens per shard | ~57.6M |
| Storage per shard | ~92 MB |


## Tokenizer

| Field | Value |
|---|---:|
| Type | Byte-level BPE |
| Vocabulary size | 16,384 |
| Tokenizer-training characters | 450M |
| Characters per token | 4.383 |
| Bytes per token | 4.390 |

## Model Architecture

| Component | Configuration |
|---|---:|
| Architecture | Decoder-only Transformer |
| Trainable parameters | 97,655,296 |
| Transformer blocks | 12 |
| Model width | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Context length | 1,024 tokens |
| Vocabulary size | 16,384 |
| Activation | GELU |
| Dropout | 0.1 |
| Positional encoding | RoPE |

RoPE does not introduce a learned positional-embedding table. The parameter count above includes the effect of tying weights for token embedding and lm_head.

## Optimization

| Hyperparameter | Value |
|---|---:|
| Optimizer | AdamW |
| Peak learning rate | 3 × 10⁻⁴ |
| Minimum learning rate | 3 × 10⁻⁵ |
| Weight decay | 0.01 |
| Adam betas | (0.9, 0.999) |
| Schedule | Cosine decay |
| Warmup | 40 optimizer steps |
| Micro-batch size | 16 sequences/GPU |
| Gradient accumulation | 32 micro-steps |
| Global batch size | 512 sequences |
| Tokens per optimizer step | 524,288 |
| Gradient clipping | None |
| Evaluation interval | 10 optimizer steps |
| Evaluation batches | 64 |
| Data-loader workers | 4 |

## Training Progress

The selected checkpoints below summarize the loss trajectory. Evaluations were run every 10 optimizer steps and once more at the final step.

| Step | Progress | Tokens processed | Train loss | Train BPB | Validation loss | Validation BPB |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.3% | 5,242,880 | 8.6661 | 2.8279 | 8.6631 | 2.8246 |
| 930 | 25.0% | 487,587,840 | 4.2534 | 1.3880 | 4.2452 | 1.3841 |
| 1,860 | 49.9% | 975,175,680 | 3.7385 | 1.2200 | 3.7380 | 1.2188 |
| 2,790 | 74.9% | 1,462,763,520 | 3.5823 | 1.1690 | 3.5837 | 1.1685 |
| **3,720** | **99.8%** | **1,950,351,360** | **3.5266** | **1.1508** | **3.5299** | **1.1509** |
| 3,726 | 100.0% | 1,953,497,088 | 3.5274 | 1.1510 | 3.5306 | 1.1512 |

Training and validation metrics remained close throughout the run, with no obvious divergence or instability in the recorded evaluations and no signs of 
overfitting have been observed.

## Compute Accounting

The run used a global batch of 512 sequences and a context length of 1,024 tokens:

```text
tokens per optimizer step = 512 × 1,024
                          = 524,288 tokens

estimated tokens processed = 3,726 × 524,288
                           = 1,953,497,088 tokens
```

Using the common dense-Transformer training approximation of `6 × parameters × tokens`:

```text
estimated training FLOPs = 6 × 97,655,296 × 1,953,497,088
                         ≈ 1.145 × 10¹⁸ FLOPs
                         ≈ 1.145 EFLOP
```


Derived runtime values:

| Derived runtime metric | Value |
|---|---:|
| Amortized wall time per optimizer step | ~20.09 s |
| Estimated end-to-end token throughput | ~26,093 tokens/s |
| Model's FLOP rate | ~15.3 TFLOP/s (4% MFU) |
| Estimated cost per million tokens | ~$0.020 |


## Hardware and Software

| Component | Configuration |
|---|---|
| GPU | 1 × NVIDIA L40S |
| GPU memory | 46,068 MiB (48 GB nominal) |
| GPU power limit | 350 W |
| CPU | AMD EPYC 7R13 |
| Allocated CPU resources | 4 logical CPUs (2 cores / 4 threads) |
| System memory | 32 GiB  |
| Operating architecture | x86_64, KVM virtual machine |
| Python | 3.12.14 |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA runtime | 12.4 |
| NVIDIA driver | 595.91.07 |
| Driver-reported CUDA compatibility | 13.2 |

## GPU Telemetry

The aggregate values below were computed from the [gpu_metrics.csv](./runs/2026-09-05_11-41-45_UTC/gpu_metrics.csv) log.

| Metric | Average | Maximum |
|---|---:|---:|
| GPU utilization | 96.40% | 100% |
| GPU utilization while ≥90% active | 99.86% | 100% |
| GPU memory used | 27,967 MiB | 27,969 MiB |
| GPU temperature | 63.02 °C | 81 °C |

Peak observed memory usage was approximately **27.31 GiB**, or **60.7%** of the GPU's capacity. 

## Artifacts

| Artifact | Location |
|---|---|
| Run directory | `experiments/exp_002/runs/2026-09-05_11-41-45_UTC/` |
| Best checkpoint | `<run-directory>/model.pt` |
| Training log | `<run-directory>/train.log` |
| GPU metrics | `<run-directory>/gpu_metrics.csv` |


## Findings

1. **Model FLOP Utilization is low.** Despite a 96.4% average GPU-utilization reading, the MFU was only 4%, indicating substantial headroom in increasing arithmetic throughput.
2. **The baseline converged successfully.** Validation loss fell from 8.6631 to 3.5299.
3. **Generalization remained stable.** Training and validation losses tracked one another closely without showing any signs of overfitting.


## Conclusion

This experiment provides a successful, compute-constrained baseline for a roughly 100M-parameter language model. It achieved a best validation BPB of **1.1509** after processing approximately **1.953B tokens** on one L40S. The loss trajectory was stable and showed no signs of overfitting, a low MFU of 4% revealed that future work should prioritize training-system efficiency.