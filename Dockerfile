ARG CUDA_VERSION=10.2
# See possible types: https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated
ARG IMAGE_TYPE=runtime

FROM nvidia/cudagl:${CUDA_VERSION}-${IMAGE_TYPE}-ubuntu18.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home

RUN apt update && \
    apt install -y gcc g++ \
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

COPY ./environment.yml ./

RUN conda env update -n base --file ./environment.yml && conda clean -ya &&  rm ./environment.yml

WORKDIR /home/app

COPY ./bfm ./3DDFA_V2/bfm

COPY ./configs ./3DDFA_V2/configs

COPY ./FaceBoxes ./3DDFA_V2/FaceBoxes

COPY ./models ./3DDFA_V2/models

COPY ./Sim3DR ./3DDFA_V2/Sim3DR

COPY ./utils ./3DDFA_V2/utils

COPY ./weights ./3DDFA_V2/weights

COPY ./demo_video_smooth.py ./demo_video.py ./demo_webcam_smooth.py ./demo.py ./extract_facelabinfo.py \
    ./latency.py ./speed_cpu.py ./TDDFA_ONNX.py ./TDDFA.py ./build.sh ./3DDFA_V2/

RUN cd ./3DDFA_V2 && \
    sh ./build.sh

WORKDIR /home/app/3DDFA_V2

FROM nvidia/cudagl:${CUDA_VERSION}-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

ARG CONDA_DIR=/opt/conda

COPY --from=builder /opt/conda /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

ARG PROJECT_DIR=/home/app/3DDFA_V2

COPY --from=builder ${PROJECT_DIR} ${PROJECT_DIR}

WORKDIR ${PROJECT_DIR}
