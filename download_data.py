from dynabench.dataset import download_equation

EQUATION = ["advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffusion", "wave"]
RESOLUTION = ["low", "medium", "high", "full"]
STRUCTURE = ["grid", "cloud"]

for eq in EQUATION:
    for struc in STRUCTURE:
        for res in RESOLUTION:
            if not (struc == "cloud" and res == "full"):
                download_equation(equation=eq, structure=struc, resolution=res)
            else:
                pass
