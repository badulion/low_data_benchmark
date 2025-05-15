import os
import subprocess
from itertools import product

SUBMIT_SCRIPT = True
WRITE_SCRIPT = True

# Parameter options
# ["grind", "feast", "cnn", "fno", "gat", "gcn", "geo_fno", "graphpde", "neuralPDE", "pointgnn", "pointnet", "ptv1", "ptv3", "resnet"]
# grind and feast need lower batchsize
models = ["grind", "feast", "gat", "gcn", "geo_fno", "graphpde", "pointgnn", "pointnet", "ptv1", "ptv3"] # all point models
equations = ["advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffusion", "wave"]
resolutions = ["low", "medium", "high", "full"]
version = '10:1'
#Normal: low: 128, medium: 128, high: 64, full: 8
#Grind/Feast: low: 32/128, medium: 16/16, high: 16/16, full: 8/8
batch = 128
wandblog = 'final'
structure = 'grid'

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
#SBATCH --mem=100G
#SBATCH --tmp=100G
#SBATCH --signal=USR1@100
#SBATCH --requeue

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

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

srun python main.py model={model} equation={equation} Structure={structure} resolution={resolution} version={version} Batch_size={batch} Epochs=10 Base_path_data=/data/42-julia-hpc-rz-lsx/s391057/dynabench wandblog={log}

sleep 60

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
    script_content = template.format(model=model, equation=equation, structure=structure, resolution=resolution, version=version, batch=batch, log=wandblog)
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