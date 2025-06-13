import wandb
import os
from tqdm import tqdm 

wandb_key = f"{os.getenv('WANDBKEY')}"
wandbloggedin = wandb.login(key = wandb_key, relogin=True)

# --- CONFIGURATION ---
ENTITY = "magnet4cardiac7t"        # Replace with your wandb username or team name
PROJECT = "dynabench2-final"       # Replace with your wandb project name

# Authenticate (make sure your WANDB_API_KEY is set in env or already logged in)
api = wandb.Api()

# Retrieve all runs for the project
runs = api.runs(f"{ENTITY}/{PROJECT}")

total_gpu_seconds = 0

# Iterate through all runs
for run in tqdm(runs, desc="Processing runs"):
    # 'gpu_seconds' is stored in the run summary if available
    # gpu_time = run.summary.get("system.gpu.gpu_seconds")
    
    # if gpu_time:
    #     total_gpu_seconds += gpu_time
    # else:
    #     print(f"Run {run.name} is missing GPU usage data.")
    history = run.history()
    metrics = run._attrs['systemMetrics']

    if 'system.gpu.0.powerPercent' in metrics and '_timestamp' in history:
        gpu_util = metrics['system.gpu.0.powerPercent']
        #print(gpu_util)
        timestamps = history['_timestamp'].array  # in seconds
        #print(timestamps)

        # Compute approximate GPU time
        import numpy as np
        delta_t = timestamps[-1] - timestamps[0]  # total time span in seconds
        #delta_t = np.diff(timestamps)  # time between records
        #util_avg = (gpu_util[:-1] + gpu_util[1:]) / 2  # average util between intervals

        # Total GPU time (in seconds of full utilization)
        gpu_time_sec = delta_t #np.sum(util_avg * delta_t) / 100.0

        #print(f"Approximate GPU time: {gpu_time_sec:.2f} seconds")
    else:
        print("GPU utilization data not found.")
    
    total_gpu_seconds += gpu_time_sec
    #print(total_gpu_seconds)

total_gpu_hours = total_gpu_seconds / 3600

print(f"\nTotal GPU Hours used in project '{PROJECT}': {total_gpu_hours:.2f} hours")