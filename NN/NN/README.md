# MASTERS

## About

Super-Resolution UNET Based NN architecture and more!


## Environment setup

### Micromamba installation

For Linux, macOS, or Git Bash on Windows install with:

`"${SHELL}" <(curl -L micro.mamba.pm/install.sh)`

For Windows Powershell:

`Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)`

For more informations, visit:
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

### Env creation
1. Create Environment (u can use pytorch_env as name or any other arbitrary name u like):

    - `micromamba create -n pytorch_env python=3.11`

    - `micromamba activate pytorch_env`

Be careful of python version you will be using, some packages might be unavailable!

2. Install packages:
    - Pytorch
        - For Windows:
            - (NVIDIA GPU) `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
            
        - For Linux:
            - (AMD GPU) `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3` 
            - (NVIDIA GPU) `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
    - CuPy
        - For Windows:
            - (NVIDIA GPU) `pip install cupy==13.6.0`

        - For Linux: 
            - (NVIDIA GPU) `pip install cupy==13.6.0`
            
    - For rest packages, please use `pyproject.toml`

3. Install <b>NN project</b> (operation should be done at level directory within pyproject.toml): `pip install .`



## Pre Commit hook installation
After cloning the repo, type (your local python version should match with language_version defined in pre commit hook):

1. Install pre-commit

    For conda:

    `> conda install -c conda-forge pre_commit`

    For micromamba:

    `> micromamba install pre-commit`

2. Install pre-commit to repository:

    `> pre-commit install`

For more informations: visit:
https://gdevops.gitlab.io/tuto_git/tools/pre-commit/articles/2018/2018.html