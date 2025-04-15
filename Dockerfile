FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y g++

ENV TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0"

WORKDIR /workspace

# rewuirements
RUN pip install numpy
RUN pip install tqdm
RUN pip install torch==2.4.0
RUN pip install torchdiffeq
RUN pip install dynabench
RUN pip install torch_geometric
RUN pip install pytorch-lightning
RUN pip install hydra-core
RUN pip install wandb
RUN pip install tensorboard
RUN pip install addict
RUN pip install spconv-cu118

RUN pip install torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu118.html


RUN python3 -c "import torch; print(torch.__version__)"