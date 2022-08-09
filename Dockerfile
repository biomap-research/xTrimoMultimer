# syntax = docker/dockerfile:1.4
FROM mambaorg/micromamba:0.24.0 AS build

FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
RUN <<eot bash
    apt-key del 7fa2af80
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
    apt-get update && apt-get install -y wget cuda-minimal-build-11-0 git openssh-server
eot

COPY --from=build /bin/micromamba /opt/conda/bin/micromamba

ENV MAMBA_ROOT_PREFIX /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN <<eot bash
    /opt/conda/bin/micromamba shell init -s bash -p /opt/conda
eot

COPY requirements /opt/xTrimoMultimer/requirements
RUN <<eot bash
    mkdir /root/.pip
    touch /root/.pip/pip.conf
    cat >> /root/.pip/pip.conf<< EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF
    mkdir /etc/conda/
    touch /etc/conda/condarc
    cat >> /etc/conda/condarc<< EOF
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
    cat /etc/conda/condarc
eot

# installing into the base environment since the docker container wont do anything other than run xtrimomultimer
RUN --mount=type=cache,target=/opt/conda/pkgs micromamba create -n xtrimomultimer -f /opt/xTrimoMultimer/requirements/environment.yaml -y  && micromamba clean --all

COPY xtrimomultimer /opt/xTrimoMultimer/xtrimomultimer
COPY scripts /opt/xTrimoMultimer/scripts
COPY tests /opt/xTrimoMultimer/tests
COPY inference.py /opt/xTrimoMultimer/inference.py
COPY setup.py /opt/xTrimoMultimer/setup.py
COPY lib/openmm.patch /opt/xTrimoMultimer/lib/openmm.patch
RUN wget -q -P /opt/xTrimoMultimer/xtrimomultimer/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
RUN patch -p0 -d /opt/conda/envs/xtrimomultimer/lib/python3.7/site-packages/ < /opt/xTrimoMultimer/lib/openmm.patch

WORKDIR /opt/xTrimoMultimer
# RUN /bin/bash -c "eval '$(micromamba shell hook --shell=bash)' && micromamba activate xtrimomultimer && python3 setup.py install"
