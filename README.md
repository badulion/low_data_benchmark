# Dynabench 2.0 - Low-Data Benchmark

## Models tested in with the Dynabench Dataset
| Model Name       | Domain           | 1-Step Loss | 16-Step Loss | Number of Parameters |
|------------------|------------------|-------------|--------------|----------------------|
| CNN              | Grid             | 0.452       | 0.389        | ~2.5M                |
| FNO              | Grid             | 0.398       | 0.342        | ~2.5M                |
| ResNet           | Grid             | 0.398       | 0.342        | ~2.5M                |
| NeuralPDE        | Grid             | 0.398       | 0.342        | ~2.5M                |
| GAT              | Cloud            | 0.000       | 0.000        | ~2.5M                |
| GCN              | Cloud            | 0.000       | 0.000        | ~2.5M                |
| Feast            | Cloud            | 0.421       | 0.365        | ~2.5M                |
| Geo FNO          | Cloud            | 0.000       | 0.000        | ~2.5M                |
| Graph PDE        | Cloud            | 0.000       | 0.000        | ~2.5M                |
| Grind            | Cloud            | 0.000       | 0.000        | ~2.5M                |
| Point GNN        | Cloud            | 0.000       | 0.000        | ~2.5M                |
| PointNet         | Cloud            | 0.000       | 0.000        | ~2.5M                |
| PT v1            | Cloud            | 0.000       | 0.000        | ~2.5M                |
| PT v3            | Cloud            | 0.000       | 0.000        | ~2.5M                |


## How to get started
```bash
# set up a python environment
python3 -m venv env

# source the environment
source env/bin/activate

# install dependencies
pip install setuptools

pip install --no-build-isolation -r requirements.txt

pip install torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```