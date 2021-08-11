ARG CUDA_VERSION=10.2
# See possible types: https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated
ARG IMAGE_TYPE=runtime

FROM nvidia/cudagl:${CUDA_VERSION}-${IMAGE_TYPE}-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home

RUN apt update && \
    apt install -y gcc \
    wget && \
    apt autoremove -y && \
    apt clean -y

# Install Miniconda
ARG CONDA_VERSION=py38_4.9.2

ARG CONDA_DIR=/opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget -q -O ./miniconda.sh http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh \
    && sh ./miniconda.sh -bfp $CONDA_DIR \
    && rm ./miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /home/app

COPY ./environment.yml ./environment.yml

RUN conda env update -n base --file ./environment.yml && conda clean -ya &&  rm ./environment.yml

WORKDIR /home/app

COPY ./ ./

RUN cd ./3DDFA_V2 && \
    sh ./build.sh

WORKDIR /home/app/3DDFA_V2