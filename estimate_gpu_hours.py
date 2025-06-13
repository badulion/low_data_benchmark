import wandb
import os
import tqdm

# --- CONFIGURATION ---
ENTITY = "magnet4cardiac7t"        # Replace with your wandb username or team name
PROJECT = "dynabench2-final"       # Replace with your wandb project name

# Authenticate (make sure your WANDB_API_KEY is set in env or already logged in)
api = wandb.Api()

# Retrieve all runs for the project
runs = api.runs(f"{ENTITY}/{PROJECT}")

total_gpu_seconds = 0

# Iterate through all runs
for run in tqdm.tqdm(runs):
    # 'gpu_seconds' is stored in the run summary if available
    gpu_time = run.summary.get("system.gpu.gpu_seconds")
    
    if gpu_time:
        total_gpu_seconds += gpu_time
    else:
        print(f"Run {run.name} is missing GPU usage data.")

total_gpu_hours = total_gpu_seconds / 3600

print(f"\nTotal GPU Hours used in project '{PROJECT}': {total_gpu_hours:.2f} hours")