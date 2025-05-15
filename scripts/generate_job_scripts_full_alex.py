import os
import subprocess
from itertools import product

SUBMIT_SCRIPT = True
WRITE_SCRIPT = True

# Parameter options
models = ["gat", "pointgnn", "grind", "feast"] #["cnn", "fno", "resnet", "neuralPDE", "gat", "gcn", "geo_fno", "graphpde", "pointgnn", "pointnet", "ptv1", "ptv3", "grind", "feast"]
equations = ["advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffusion", "wave"]
resolutions = ["full"]
version = '0:2'
batch = 8

# Directory to save job scripts
output_dir = "sbatch_jobs"
os.makedirs(output_dir, exist_ok=True)

# Template for the SLURM script
template = """#!/bin/bash -l
#
#SBATCH --output=job_out/output_{model}_{equation}_{resolution}_{version}.log
#SBATCH --job-name=db2-{model}_{equation}_{resolution}_{version}
#SBATCH --export=NONE
#SBATCH --gres=gpu:a40:1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@100
#SBATCH --requeue

unset SLURM_EXPORT_ENV

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd ~/DB2_newdb

mkdir -p /home/vault/b190cb/b190cb18/db2_output/status

echo "Copying files to TMP..."

mkdir -p $TMPDIR/data/{equation}/grid/{resolution}
cp -r "/home/atuin/b190cb/b190cb18/data/{equation}/grid/{resolution}" "$TMPDIR/data/{equation}/grid"

echo "Done copying files to TMP\n"

source env/bin/activate

# Auto-requeue handler
requeue_job() {{
    echo "Requeuing job..."
    # touch output/status/{model}:{equation}:{resolution}:{version}:requeue.txt
}}
trap requeue_job USR1

# Path to status file created when training finishes
STATUS_FILE=/home/vault/b190cb/b190cb18/db2_output/status/{model}:{equation}:{resolution}:{version}.txt

srun python main.py model={model} equation={equation} resolution={resolution} version={version} Batch_size={batch} Epochs=10 Base_path_data=$TMPDIR/data output_dir=/home/vault/b190cb/b190cb18/db2_output

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
