# MASTERS

## About

Super-Resolution UNET Based NN architecture and more!

See NN/ folder for further information!

## Environment setup

### Micromamba installation

For Linux, macOS, or Git Bash on Windows install with:

`"${SHELL}" <(curl -L micro.mamba.pm/install.sh)`

For Windows Powershell:

`Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)`

For more informations, visit:
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

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