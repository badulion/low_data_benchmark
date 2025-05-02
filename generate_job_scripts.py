import os
import subprocess
from itertools import product

SUBMIT_SCRIPT = True
WRITE_SCRIPT = True

# Parameter options
models = ["grind", "feast", "cnn", "fno", "gat", "gcn", "geo_fno", "graphpde", "neuralPDE", "pointgnn", "pointnet", "ptv1", "ptv3", "resnet"]
equations = ["advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffusion", "wave"]
resolutions = ["high", "medium", "high", "full"]
version = '0:1'
batch = 128

# Directory to save job scripts
output_dir = "sbatch_jobs"
os.makedirs(output_dir, exist_ok=True)

# Template for the SLURM script
template = """#!/bin/bash
#SBATCH -J Dynabench2_{model}_{equation}_{resolution}_{version}
#SBATCH --output=job_out/output_{model}_{equation}_{resolution}_{version}.log
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=20G
#SBATCH --tmp=20G
#SBATCH --signal=USR1@100
#SBATCH --requeue

cd ~/dynabench2

mkdir -p output/status

source env/bin/activate

# Auto-requeue handler
requeue_job() {{
    echo "Requeuing job..."
    touch output/status/{model}:{equation}:{resolution}:{version}:requeue.txt
}}
trap requeue_job USR1

# Path to status file created when training finishes
STATUS_FILE=output/status/{model}:{equation}:{resolution}:{version}.txt

srun python main.py MODEL={model} equation={equation} resolution={resolution} version={version} Batch_size={batch} Epochs=10 Base_path_data=/data/42-julia-hpc-rz-lsx/s391057/dynabench

sleep 10

# Requeue if training isn't finished
if [[ -f "$STATUS_FILE" ]] && grep -q "TRAINING_COMPLETED" "$STATUS_FILE"; then
    echo "Training completed. Not requeuing."
else
    echo "Training not complete. Requeuing..."
    scontrol requeue $SLURM_JOB_ID
fi
"""

# Generate job scripts
for model, equation, resolution in product(models, equations, resolutions):
    script_content = template.format(model=model, equation=equation, resolution=resolution, version=version, batch=batch)
    filename = f"job_{model}_{equation}_{resolution}.sh"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(script_content)

    if SUBMIT_SCRIPT:
        try:
            result = subprocess.run(["sbatch", filepath], capture_output=True, text=True, check=True)
            print(f"Submitted: {filepath} -> {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting {filepath}: {e.stderr.strip()}")

print(f"Generated {len(models) * len(equations) * len(resolutions)} job scripts in '{output_dir}'")