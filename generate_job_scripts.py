import os
from itertools import product

# Parameter options
models = ["cnn", "feast", "fno", "gat", "gcn", "geo_fno", "graphpde", "grind", "neuralPDE", "pointgnn", "pointnet", "ptv1", "ptv3", "resnet"]
equations = ["advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffusion", "wave"]
resolutions = ["low", "medium", "high", "full"]

# Directory to save job scripts
output_dir = "sbatch_jobs"
os.makedirs(output_dir, exist_ok=True)

# Template for the SLURM script
template = """#!/bin/bash
#SBATCH -J Dynabench2
#SBATCH --output=job_out/output_{model}_{equation}_{resolution}.log
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=20G
#SBATCH --tmp=20G

cd ~/dynabench2

source env/bin/activate

srun python main.py MODEL={model} equation={equation} resolution={resolution} Batch_size=32 Epochs=10 Base_path_data=/data/42-julia-hpc-rz-lsx/s391057/dynabench
"""

# Generate job scripts
for model, equation, resolution in product(models, equations, resolutions):
    script_content = template.format(model=model, equation=equation, resolution=resolution)
    filename = f"job_{model}_{equation}_{resolution}.sh"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(script_content)

print(f"Generated {len(models) * len(equations) * len(resolutions)} job scripts in '{output_dir}'")