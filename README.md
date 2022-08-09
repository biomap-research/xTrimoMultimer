# xTrimoMultimer

![](https://img.shields.io/badge/Made%20with-ColossalAI-blueviolet?style=flat)
![](https://img.shields.io/github/v/release/biomap-research/xTrimoMultimer)
[![GitHub license](https://img.shields.io/github/license/biomap-research/xTrimoMultimer)](https://github.com/biomap-research/xTrimoMultimer/blob/main/LICENSE)

Optimizing Protein Structure Prediction Model for both monomer and multimer on GPU Clusters

xTrimoMultimer is a cooperation project by [Biomap](https://www.biomap.com/en/) and [HPC-AI TECH](https://www.hpcaitech.com) which provides a **high-performance implementation of AlphaFold and AlphaFold Multimer** with the following characteristics.

1. Fast kernel performance on GPU platform.
2. Supporting Various Parallelism including Dynamic Axial Parallelism(DAP) by [FastFold](https://github.com/hpcaitech/FastFold) in multi-GPU environment for both AlphaFold monomer and multimer.
3. Support long sequence training(To be supported in the future) and inference in both monomer and multimer.

## Quick start

We strongly recommend users follow this installation manual step by step.

Choosing the way you feel comfortable between container environment and non-container environment. We have provide both way for users to make choice.

## Installation

### With existing conda

Create the virtual environment by the `environment.yaml` we provide:

```bash
conda create -n xtrimomultimer -y requirements/environment.yaml
```

Noticed that one of the dependencies `colossalai` may need CUDA Toolkit to be correctly installed. Installation under machine without NVIDIA GPU and CUDA thus may fail.

We will add a further switch to handle this problem.

#### Patch openmm (Please change the `[PATH_TO_ENV]` to the envs path):

```bash
pushd [PATH_TO_ENV]/lib/python3.7/site-packages/ && patch -p0 < ./lib/openmm.patch && popd
```

#### Activate the environment

```bash
conda activate xtrimomultimer
```

#### Compile the acceleration modules

Execute the following commands to compile the acceleration modules after first activation of your environment.

```bash
python setup.py install
```

### With Docker

To use container for reducing the influence of difference system package/configuration, a Dockerfile is provided. Since this dockerfile is written in version 1.4 of Dockerfile syntax version, we need a extention provided by the Docker officially. If you cannot find the command `docker buildx`, a detailed manual installation guide on this extension can be found in [this webpage](https://docs.docker.com/build/buildx/install/).

The building command for the container is:

```
docker buildx build . --file Dockerfile --tag extrememultimer
```

## Usage

### Development

In general, to avoid affect of different hardware/system package, we strongly recommended to develop in a virtual machine or container.

There are also some pakcage useful during development that can be installed through `pip`:

```bash
pip install -r requirements/dev.txt
```

#### Before PR

In order to make the codes in this repo be consistent and easy to read. We use `pre-commit` to manage the format issues all around the project codes. We strongly recommend to execute the following command under the root directory of project after installing packages in `Development` section:

```bash
pre-commit run --all-files
```

### Inference

Please specify the `cuda_device`, `fasta_paths` and `output_dir`, then execute the following bash,
all the results will be saved in `output_dir`.

Use the following script to see the usage of `inference.py`

```bash
python inference.py --help
```

A sample running script has been put into `bin/inference.sh` for reference.

### Test

#### Prepare

To running all the test cases, you need to install extra packages to install test environ requirements.

Use the following command to install the test requirements:

```bash
pip install -r requirements/test.txt
```

#### Running Test Cases

To running all the test cases, execute the following command under the vritual environment:

```
# Add -v for verbose mode
pytest [the_path_to_special_test_file] [-v]
```

Or with extra `coverage` command to generate code coverage report:

```bash
coverage run -m pytest [the_path_to_special_test_file]
coverage report -m
```

Indicators in `[]` can be ignored or deleted to run all the test cases.

## Copyright Notice

Caution: To deprecate our inner feature processing logics, OpenFold's data processing code has been greatly introduced before feature_processing parts. This may cause some problems, because the open source version of OpenFold's multimer has not yet published with comprehensive test. If there exists any problem on data processing parts, please feel free to leave a message on [Issues](https://github.com/biomap-research/xTrimoMultimer/issues) page.

AlphaFold's, OpenFold's and, by extension, xTrimoMultimer source code is licensed under the permissive Apache Licence, Version 2.0.

## Reference

- [OpenFold, Ahdritz, Gustaf and Bouatta .etc, 2021](https://github.com/aqlaboratory/openfold/)

- [FastFold, Shenggan Cheng, Ruidong Wu and Zhongming Yu .etc](https://github.com/hpcaitech/FastFold)
