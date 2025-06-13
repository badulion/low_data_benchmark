#!/bin/bash
#SBATCH -J DB2_{model}_{equation}_{resolution}_{version}
#SBATCH --output={output_path}/job_out/output_{model}_{equation}_{resolution}_{version}.log
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=100G
#SBATCH --tmp=100G
#SBATCH --signal=USR1@100
#SBATCH --requeue

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd /home/s391057/low_data_benchmark

mkdir -p {output_path}

# Path to status file created when training finishes
STATUS_FILE={output_path}/status/{model}:{equation}:{resolution}:{version}.txt

if [[ -f "$STATUS_FILE" ]] && grep -q "TRAINING_COMPLETED" "$STATUS_FILE"; then
    echo "Training completed. Not starting."

else

    # activate env
    source /home/s391057/low_data_benchmark/env/bin/activate

    # Auto-requeue handler
    requeue_job() {{
        echo "Requeuing job..."
        touch {output_path}/status/{model}:{equation}:{resolution}:{version}:requeue.txt
    }}
    trap requeue_job USR1

    # start job
    srun python main.py model={model} equation={equation} resolution={resolution} version={version} Batch_size={batch} Epochs={epochs} Base_path_data=/data/42-julia-hpc-rz-lsx/s391057/dynabench wandblog={log} output_dir={output_path}

    sleep 10

    # Requeue if training isn't finished
    if [[ -f "$STATUS_FILE" ]] && grep -q "TRAINING_COMPLETED" "$STATUS_FILE"; then
        echo "Training completed. Not requeuing."
    else
        echo "Training not complete. Requeuing..."
        scontrol requeue $SLURM_JOB_ID
    fi

fi