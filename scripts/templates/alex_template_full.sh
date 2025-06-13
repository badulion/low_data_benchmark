#!/bin/bash -l
#
#SBATCH --output={output_path}/job_out/output_{model}_{equation}_{resolution}_{version}.log
#SBATCH --job-name=DB2-{model}_{equation}_{resolution}_{version}
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

cd ~/low_data_benchmark

mkdir -p {output_path}/status

# Path to status file created when training finishes
STATUS_FILE={output_path}/status/{model}:{equation}:{resolution}:{version}.txt

if [[ -f "$STATUS_FILE" ]] && grep -q "TRAINING_COMPLETED" "$STATUS_FILE"; then
    echo "Training completed. Not starting."

else
    # Copy files
    echo "Copying files to TMP..."
    mkdir -p $TMPDIR/data/{equation}/grid/{resolution}
    cp -r "/home/atuin/b190cb/b190cb18/data/{equation}/grid/{resolution}" "$TMPDIR/data/{equation}/grid"
    echo "Done copying files to TMP\n"
    
    # activate env
    source ~/low_data_benchmark/env/bin/activate

    # Auto-requeue handler
    requeue_job() {{
        echo "Requeuing job..."
        # touch output/status/{model}:{equation}:{resolution}:{version}:requeue.txt
    }}
    trap requeue_job USR1
    
    # start job
    srun python main.py model={model} equation={equation} resolution={resolution} Structure=grid version={version} Batch_size={batch} Epochs={epochs} val_batches=0.05 ValCheckInterval=0.25 Base_path_data=$TMPDIR/data output_dir={output_path}

    sleep 10

    # Requeue if training isn't finished
    if [[ -f "$STATUS_FILE" ]] && grep -q "TRAINING_COMPLETED" "$STATUS_FILE"; then
        echo "Training completed. Not requeuing."
    else
        echo "Training not complete. Requeuing..."
        scontrol requeue $SLURM_JOB_ID
    fi
    
fi