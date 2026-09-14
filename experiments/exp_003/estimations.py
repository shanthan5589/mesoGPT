'''
Peak and training_compute_budget should be same units.

You can calculate metrics for both including and excluding eval batches. 
If you want to exclude eval batches, set eval_interval to 0. 
If you want to include eval batches, set eval_interval to a non-zero value.

'''

import math

peak_gpu_performance = 362.05
number_of_trainable_params = 97655296
observed_time_taken_to_complete_one_optimizer_step = 17.16
observed_time_to_compute_eval_loss = 0
eval_interval = 0

global_batch_size = 512
context_length = 1024
d_model = 768

training_token_budget = 20 * number_of_trainable_params
training_compute_budget = (6 * number_of_trainable_params * training_token_budget) / (10 ** 12)  # In TFLOPS

optimizer_steps = math.ceil(training_token_budget / (global_batch_size * context_length))

# Recompute training_token_budget and training_compute_budget based on optimizer_steps.
training_token_budget = optimizer_steps * global_batch_size * context_length
training_compute_budget =  (6 * number_of_trainable_params * training_token_budget) / (10 ** 12)  # In TFLOPS


if eval_interval == 0:
    total_eval_batches = 0
else:
    total_eval_batches = (optimizer_steps // eval_interval) if optimizer_steps % eval_interval == 0 else (optimizer_steps // eval_interval) + 1


compute_per_optimizer_step = training_compute_budget / optimizer_steps


observed_perf = compute_per_optimizer_step / observed_time_taken_to_complete_one_optimizer_step
observed_mfu = (observed_perf / peak_gpu_performance) 

calc_time_to_train = (training_compute_budget) / (peak_gpu_performance * observed_mfu * 3600)
time_spent_on_eval_batches = (total_eval_batches * observed_time_to_compute_eval_loss) / 3600

total_time = calc_time_to_train + time_spent_on_eval_batches 


training_throughput = training_token_budget / (total_time * 3600)
flop_throughput = training_compute_budget / (total_time * 3600)


print(f"Observed compute Throughput: {flop_throughput:.3f} TFLOP")
print(f"Observed token throughput: {training_throughput:.1f} tokens")
print(f"Observed MFU: {observed_mfu:.4f}")


print(f"Calculated time to train (train + eval): {total_time:.3f} hrs")
print(f"Calculated time to train (train): {calc_time_to_train:.3f} hrs")