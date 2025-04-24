from dynabench.dataset import download_equation

EQUATION = ["advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffusion", "wave"]
RESOLUTION = ["low", "medium", "high", "full"]
STRUCTURE = ["grid", "cloud"]

for eq in EQUATION:
    for struc in STRUCTURE:
        for res in RESOLUTION:
            try:
                download_equation(equation=eq, structure=struc, resolution=res, data_dir="/data/42-julia-hpc-rz-lsx/s391057/dynabench", tmp_dir="/data/42-julia-hpc-rz-lsx/s391057/dynabench/tmp")
            except:
                print(f'Could not download: {eq} {struc} {res}')
