# Training Compute, Runtime, GPU Utilization, and Cost Estimates

This document calculates the training requirements in a fixed order:

1. derive the training-token budget from the model size
2. derive the number of optimizer updates from the token budget, global batch size, and context length
3. derive the total training-compute budget
4. estimate runtime and rental cost using an assumed model FLOPs utilization (MFU)
5. measure optimizer-update and evaluation durations on the rented GPU
6. derive the achieved GPU performance and MFU from those measurements
7. recalculate the expected runtime and rental cost using the measured performance

## 1. Derive the training token budget

### 1.1 Model size

The model contains:

$$
N = 97{,}655{,}296\ \text{trainable parameters}
$$

### 1.2 Training-token budget

Following token per parameter ratio from the Chinchilla Scaling Laws, we are going to train each
parameter with 20 tokens of data. Therefore, the training-token budget is:

$$
D = 20N
$$

Substitute the model size:

$$
\begin{aligned}
D
&= 20 \times 97{,}655{,}296 \\
&= 1{,}953{,}105{,}920\ \text{tokens} \\
&\approx 1.953\ \text{billion tokens}
\end{aligned}
$$

The planned training-token budget is therefore approximately **1.95 billion tokens**.

### 1.3 Number of optimizer updates

The global batch size is 512 sequences, and each sequence contains 1,024 tokens:

$$
B_{\mathrm{global}} = 512\ \text{sequences}
$$

$$
L_{\mathrm{context}} = 1024\ \text{tokens/sequence}
$$

Therefore, the number of training tokens processed by one optimizer update is:

$$
\begin{aligned}
D_{\mathrm{update}}
&= B_{\mathrm{global}} \times L_{\mathrm{context}} \\
&= 512 \times 1024 \\
&= 524{,}288\ \text{tokens/update}
\end{aligned}
$$

Divide the training-token budget by the number of tokens processed per optimizer update:

$$
\begin{aligned}
N_{\mathrm{updates}}
&= \left\lceil \frac{D}{D_{\mathrm{update}}} \right\rceil \\
&= \left\lceil
\frac{1{,}953{,}105{,}920}{524{,}288}
\right\rceil \\
&= \left\lceil 3725.254 \right\rceil \\
&= \boxed{3726\ \text{optimizer updates}}
\end{aligned}
$$

The result is rounded up because 3,725 complete optimizer updates would not process the entire training-token budget.

### 1.4 Total training-compute budget

For a dense Transformer, approximate the training compute as:

$$
C_{\mathrm{train}} \approx 6ND
$$

where:

- $N$ is the number of trainable parameters;
- $D$ is the number of training tokens;
- the factor 6 approximates the forward-pass and backward-pass operations used during training.

Substitute the parameter count and token budget:

$$
\begin{aligned}
C_{\mathrm{train}}
&\approx 6 \times 97{,}655{,}296 \times 1{,}953{,}105{,}920 \\
&= 1.144386820421714 \times 10^{18}\ \text{FLOP} \\
&\approx 1.144 \times 10^{18}\ \text{FLOP}
\end{aligned}
$$

### 1.5 Convert FLOP to TFLOP

The maximum theoretical GPU performance is expressed in TFLOP/s, so the training-compute budget must be converted to TFLOP before it is used with that performance figure.

$$
1\ \text{TFLOP} = 10^{12}\ \text{FLOP}
$$

Therefore:

$$
\begin{aligned}
C_{\mathrm{train}}
&= \frac{1.144 \times 10^{18}\ \text{FLOP}}
{10^{12}\ \text{FLOP/TFLOP}} \\
&= 1.144 \times 10^6\ \text{TFLOP}
\end{aligned}
$$

The remaining calculations use:

$$
\boxed{C_{\mathrm{train}} = 1.144 \times 10^6\ \text{TFLOP}}
$$

> FLOP is a total number of floating-point operations. FLOP/s is a processing rate. Dividing FLOP by FLOP/s produces seconds.

## 2. Establish the known values and planning assumptions

### 2.1 Known values

| Quantity | Symbol | Value |
|---|---:|---:|
| Maximum theoretical GPU performance | $P_{\mathrm{GPU,max}}$ | $362.05\ \text{TFLOP/s}$ |
| GPU rental price | $R_{\mathrm{GPU}}$ | $\$1.861/\text{hour}$ |
| Total training-compute budget | $C_{\mathrm{train}}$ | $1.144 \times 10^6\ \text{TFLOP}$ |
| Total number of optimizer updates | $N_{\mathrm{updates}}$ | $3{,}726$ |
| Evaluation cadence | $K_{\mathrm{eval}}$ |  Run evaluation after every 10 completed optimizer steps |

### 2.2 Planning assumptions

Before renting and measuring the actual GPU utilization, assume:

| Assumption | Symbol | Value |
|----|----:|----:|
| Model FLOPs utilization | $U_{\mathrm{assumed}}$ | $0.10 = 10\%$ |
| Estimated duration of one evaluation run over all 128 batches | $\tau_{\mathrm{eval,assumed}}$ | $20.00\ \text{s}$ |

Training evaluates loss once after every 10 completed optimizer updates. Each time evaluation starts, it calculates loss over:

- 64 batches from the training split; and
- 64 batches from the validation split.

Therefore, each evaluation run processes 128 batches in total. In other words:

- **10 optimizer updates** determines the frequency of evaluation; and
- **128 batches** is the evaluation work performed in each evaluation run.

## 3. Determine how many times evaluation runs

The training loop evaluates after every 10 completed optimizer updates:

$$
10, 20, 30, \ldots, 3720
$$

The number of regularly scheduled evaluation runs is:

$$
\begin{aligned}
N_{\mathrm{regular\ evaluations}}
&= \left\lfloor \frac{3726}{10} \right\rfloor \\
&= 372
\end{aligned}
$$

There are six optimizer updates after the last regular evaluation:

$$
3726 \bmod 10 = 6
$$

The training code also evaluates when the final optimizer update is reached. It therefore performs one additional evaluation run at update 3726:

$$
\begin{aligned}
N_{\mathrm{evaluations}}
&= 372 + 1 \\
&= \boxed{373\ \text{evaluation runs}}
\end{aligned}
$$

Across the entire run, this corresponds to:

$$
373 \times 128 = 47{,}744\ \text{evaluated batches}
$$

Both the assumed and measured evaluation durations cover all 128 batches in one evaluation run. Total evaluation time is therefore the duration of one evaluation run multiplied by 373.

## 4. Calculate the planned runtime using the assumed MFU

### Step 1: Calculate training compute per optimizer update

Distribute the total training-compute budget across all optimizer updates:

$$
\begin{aligned}
C_{\mathrm{update}}
&= \frac{C_{\mathrm{train}}}{N_{\mathrm{updates}}} \\
&= \frac{1.144 \times 10^6\ \text{TFLOP}}{3726} \\
&= 307.032\ \text{TFLOP/update}
\end{aligned}
$$

### Step 2: Calculate the assumed effective GPU performance

MFU is the fraction of the GPU's maximum theoretical performance that is effectively used by model training:

$$
P_{\mathrm{effective,assumed}}
= P_{\mathrm{GPU,max}} \times U_{\mathrm{assumed}}
$$

Substitute the assumed MFU:

$$
\begin{aligned}
P_{\mathrm{effective,assumed}}
&= 362.05\ \text{TFLOP/s} \times 0.10 \\
&= 36.205\ \text{TFLOP/s}
\end{aligned}
$$

### Step 3: Estimate the duration of one optimizer update

$$
\begin{aligned}
\tau_{\mathrm{update,assumed}}
&= \frac{C_{\mathrm{update}}}{P_{\mathrm{effective,assumed}}} \\
&= \frac{307.032\ \text{TFLOP}}
{36.205\ \text{TFLOP/s}} \\
&= 8.480\ \text{s/update}
\end{aligned}
$$

### Step 4: Estimate training time excluding evaluation runs

This is the time required to perform all 3,726 optimizer updates. It does not include the periodic evaluation runs, which are added in the next step. Divide total training compute by effective GPU performance:

$$
T_{\mathrm{train,assumed}}
= \frac{C_{\mathrm{train}}}
{P_{\mathrm{GPU,max}} \times U_{\mathrm{assumed}}}
$$

This first produces seconds:

$$
\begin{aligned}
T_{\mathrm{train,assumed}}
&= \frac{1.144 \times 10^6\ \text{TFLOP}}
{362.05\ \text{TFLOP/s} \times 0.10} \\
&= 31{,}597.846\ \text{s}
\end{aligned}
$$

Convert seconds to hours:

$$
\begin{aligned}
T_{\mathrm{train,assumed}}
&= \frac{31{,}597.846}{3600} \\
&= 8.777\ \text{hours}
\end{aligned}
$$

Equivalently, the complete hours formula is:

$$
T_{\mathrm{train,assumed\ (hours)}}
= \frac{C_{\mathrm{train}}}
{P_{\mathrm{GPU,max}} \times U_{\mathrm{assumed}} \times 3600}
$$

### Step 5: Estimate total evaluation time

The assumed 20 seconds is the duration of one complete 128-batch evaluation cycle:

$$
\begin{aligned}
T_{\mathrm{eval,assumed}}
&= N_{\mathrm{evaluations}} \times \tau_{\mathrm{eval,assumed}} \\
&= 373 \times 20.00\ \text{s} \\
&= 7{,}460\ \text{s} \\
&= \frac{7{,}460}{3600} \\
&= 2.072\ \text{hours}
\end{aligned}
$$

### Step 6: Estimate total runtime

$$
\begin{aligned}
T_{\mathrm{total,assumed}}
&= T_{\mathrm{train,assumed}} + T_{\mathrm{eval,assumed}} \\
&= 8.777 + 2.072 \\
&= \boxed{10.849\ \text{hours}}
\end{aligned}
$$

### Step 7: Estimate the GPU rental cost

$$
\begin{aligned}
\mathrm{Cost}_{\mathrm{assumed}}
&= T_{\mathrm{total,assumed}} \times R_{\mathrm{GPU}} \\
&= 10.849\ \text{hours} \times \$1.861/\text{hour} \\
&= \boxed{\$20.19}
\end{aligned}
$$

The planning estimate is therefore **10.849 hours** and **$20.19**.

## 5. Measure performance on the rented GPU

The above planning estimates depend on the assumed model FLOPs utilization (MFU) and the estimated time required to evaluate 128 batches.

To measure actual performance, rent the type of GPU you are planning to use for training your model and use a performance timer to record the duration of one optimizer step and one complete evaluation run over all 128 batches. Before recording these measurements, perform a few warm-up optimizer steps and evaluation runs. Because the goal is only to benchmark performance, running the full training loop is unnecessary. 

Observed durations are as follows:


| Measured quantity | Symbol | Value |
|---|---:|---:|
| Duration of one complete optimizer update | $\tau_{\mathrm{update,measured}}$ | $17.16\ \text{s}$ |
| Duration of one evaluation run over all 128 batches | $\tau_{\mathrm{eval,measured}}$ | $23.83\ \text{s}$ |

The optimizer-update measurement covers the full update, including all gradient-accumulation micro-batches, backward passes, and the optimizer operation. The evaluation measurement covers one complete pass through the configured 64 training-loss batches and 64 validation-loss batches.

## 6. Derive MFU and runtime from the measurements

### Step 1: Calculate the measured effective GPU performance

The effective performance is the compute assigned to one optimizer update divided by its measured duration:

$$
\begin{aligned}
P_{\mathrm{effective,measured}}
&= \frac{C_{\mathrm{update}}}{\tau_{\mathrm{update,measured}}} \\
&= \frac{307.032\ \text{TFLOP}}{17.16\ \text{s}} \\
&= 17.892\ \text{TFLOP/s}
\end{aligned}
$$

### Step 2: Derive the measured MFU

MFU is effective GPU performance divided by maximum theoretical GPU performance:

$$
U_{\mathrm{measured}}
= \frac{P_{\mathrm{effective,measured}}}{P_{\mathrm{GPU,max}}}
$$

Substitute the measured effective performance:

$$
\begin{aligned}
U_{\mathrm{measured}}
&= \frac{17.892}{362.05} \\
&= 0.0494194 \\
&= \boxed{4.942\%}
\end{aligned}
$$

### Step 3: Calculate training time excluding evaluation runs

Use the same optimizer-update time formula, replacing the assumed MFU with the measured MFU. Evaluation time is still excluded here and will be added in the next step:

$$
T_{\mathrm{train,measured\ (hours)}}
= \frac{C_{\mathrm{train}}}
{P_{\mathrm{GPU,max}} \times U_{\mathrm{measured}} \times 3600}
$$

$$
\begin{aligned}
T_{\mathrm{train,measured}}
&= \frac{1.144 \times 10^6}
{362.05 \times 0.0494194 \times 3600} \\
&= 17.761\ \text{hours}
\end{aligned}
$$

This result can be checked directly using the measured optimizer-update duration:

$$
\begin{aligned}
T_{\mathrm{train,measured}}
&= \frac{N_{\mathrm{updates}} \times \tau_{\mathrm{update,measured}}}{3600} \\
&= \frac{3726 \times 17.16}{3600} \\
&= 17.761\ \text{hours}
\end{aligned}
$$

### Step 4: Calculate total measured evaluation time

The measured 23.83 seconds covers one complete 128-batch evaluation cycle:

$$
\begin{aligned}
T_{\mathrm{eval,measured}}
&= N_{\mathrm{evaluations}} \times \tau_{\mathrm{eval,measured}} \\
&= 373 \times 23.83\ \text{s} \\
&= 8{,}888.59\ \text{s} \\
&= \frac{8{,}888.59}{3600} \\
&= 2.469\ \text{hours}
\end{aligned}
$$

### Step 5: Calculate total measured runtime

$$
\begin{aligned}
T_{\mathrm{total,measured}}
&= T_{\mathrm{train,measured}} + T_{\mathrm{eval,measured}} \\
&= 17.761 + 2.469 \\
&= \boxed{20.230\ \text{hours}}
\end{aligned}
$$

### Step 6: Calculate the measured-performance rental cost

$$
\begin{aligned}
\mathrm{Cost}_{\mathrm{measured}}
&= T_{\mathrm{total,measured}} \times R_{\mathrm{GPU}} \\
&= 20.230\ \text{hours} \times \$1.861/\text{hour} \\
&= \boxed{\$37.65}
\end{aligned}
$$

The measurement-based estimate is therefore **20.230 hours** and **$37.65**.

## 7. Final comparison

| Metric | Planning estimate | Measurement-based estimate |
|---|---:|---:|
| Model FLOPs utilization | $10.000\%$ assumed | $4.942\%$ measured |
| Effective GPU performance | $36.205\ \text{TFLOP/s}$ | $17.892\ \text{TFLOP/s}$ |
| Duration of one optimizer update | $8.480\ \text{s}$ estimated | $17.160\ \text{s}$ measured |
| Training time excluding evaluation runs | $8.777\ \text{hours}$ | $17.761\ \text{hours}$ |
| Duration of one evaluation run over all 128 batches | $20.00\ \text{s}$ assumed | $23.83\ \text{s}$ measured |
| Total evaluation time | $2.072\ \text{hours}$ | $2.469\ \text{hours}$ |
| Total runtime | **$10.849\ \text{hours}$** | **$20.230\ \text{hours}$** |
| Total GPU rental cost | **$\$20.19$** | **$\$37.65$** |

As expected, the calculated training time and cost are higher than the planning estimates. The measured MFU was lower than the assumed MFU, which led to a longer training duration and higher rental cost.


## Actual Training Results

After the complete training is completed, the measured end-to-end results were:
- Total time for training: 20.796 hours (~21 hours)
- Cost of training: $39.08


## 8. Notes:

- The GPU is billed continuously during training and evaluation.
- The total training-compute budget excludes evaluation compute. Evaluation overhead is added separately using wall-clock duration.
- The 20-tokens-per-parameter ratio is inspired by the Chinchilla Scaling Laws.
- The total training compute is is assumed to be distributed equally across the 3,726 optimizer steps.
- The measured optimizer step duration is assumed to be same for all optimizer steps.
- The measured duration of one evaluation run is assumed to be same for all 373 evaluation runs.
- The maximum theoretical GPU performance must correspond to the numerical precision used during training. For example, if the model is trained using FP16 or BF16, use the GPU’s theoretical FP16 or BF16 performance, not its FP32 performance.