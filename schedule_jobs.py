import os
import subprocess

# Directory containing the job files
job_dir = "sbatch_jobs"

# Ensure the directory exists
if not os.path.isdir(job_dir):
    raise FileNotFoundError(f"Directory '{job_dir}' not found.")

# List all files in the directory
job_files = os.listdir(job_dir)

# Filter only .sh files with 'low' resolution
low_res_jobs = [
    f for f in job_files
    if f.endswith(".sh") and f.split("_")[-1].strip(".sh") == "low"
]

# Submit each job using sbatch
for job in low_res_jobs:
    job_path = os.path.join(job_dir, job)
    try:
        result = subprocess.run(["sbatch", job_path], capture_output=True, text=True, check=True)
        print(f"Submitted: {job_path} -> {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting {job_path}: {e.stderr.strip()}")
