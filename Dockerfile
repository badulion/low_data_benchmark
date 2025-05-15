FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y g++

ENV TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0"

WORKDIR /workspace

# rewuirements
RUN pip install setuptools

# copy requirements
COPY requirements.txt ./
RUN pip install --no-build-isolation -r requirements.txt

RUN pip install torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html


RUN python3 -c "import torch; print(torch.__version__)"