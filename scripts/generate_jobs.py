import os
import subprocess
from itertools import product
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Generate job scripts
def generate_job_script(template, model, batch, epoch, equation, resolution, structure, wandblog, version, output_path):
            
    if equation == "A":
        equation = "advection"
    if equation == "B":
        equation = "burgers"
    if equation == "GD":
        equation = "gasdynamics"
    if equation == "KS":
        equation = "kuramotosivashinsky"
    if equation == "RD":
        equation = "reactiondiffusion"
    if equation == "W":
        equation = "wave"

    script_content = template.format(
        output_path=output_path,
        model=model,
        equation=equation,
        resolution=resolution,
        structure=structure,
        version=version,
        batch=batch,
        epochs=epoch,
        log=wandblog
        )

    filename = f"job_{model}_{equation}_{resolution}_{version}.sh"
    filepath = os.path.join(f"{output_path}/job_scripts", filename)

    with open(filepath, "w") as f:
        f.write(script_content)
    
    print(f"Generated job script {filepath} in '{f"{output_path}/job_scripts"}'")
    
    return filepath




@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    template_file = cfg.template_file
    models = cfg.models
    resolutions = cfg.resolutions
    structure = cfg.structure
    equations = cfg.equations
    wandblog = cfg.wandblog
    version = cfg.version
    submit_scripts = cfg.submit_scripts
    output_path = cfg.output_path
    batches = cfg.batches
    epochs = cfg.epochs

    with open(f"templates/{template_file}", "r") as file:
        template = file.read()

    # Directory to save job scripts
    os.makedirs(f"{output_path}/job_scripts", exist_ok=True)

    # Generate job scripts for all combinations of models, equations, resolutions
    for model, equation, resolution in product(models, equations, resolutions):
        filepath = generate_job_script(template, model, batches, epochs, equation, resolution, structure, wandblog, version, output_path)
        
        # directly schedule the job 
        if submit_scripts:
            try:
                result = subprocess.run(["sbatch", filepath], capture_output=True, text=True, check=True)
                print(f"Submitted: {filepath} -> {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting {filepath}: {e.stderr.strip()}")


if __name__ == "__main__":
    main()