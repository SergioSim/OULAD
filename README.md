# Analysis of the OULAD dataset

This project restarts with a new ambition:

> Explore various analytics approaches that has been applied to the OULAD
> dataset.

The previous version of this project is available at the
[old-master](https://github.com/SergioSim/OULAD/tree/old-master)
branch.

## Running this project

To run this project:

1. make sure you meet the [requirements](#requirements)
2, build the project using [docker-compose](#building-with-docker-compose-default) or
   [python virtual environment](#building-with-python-virtual-environment)

### Requirements

We use:

- [GNU make](https://www.gnu.org/software/make/) to bootstrap this project.
- [GNU wget](https://www.gnu.org/software/wget/) to download the OULAD dataset from
  [Open University](https://analyse.kmi.open.ac.uk/open_dataset).
- [unzip](https://infozip.sourceforge.net/UnZip.html) to extract the .zip compressed OULAD dataset.
- [docker compose](https://docs.docker.com/compose/) or
  [python -m venv](https://docs.python.org/3/library/venv.html) for the development environment.

### Building with docker compose (default)

1. Clone this project and run the `build` Makefile target from the root of the project (where this README.md file is located):
   ```bash
   $ make build
   ```
   This should setup the project, build the docker image and download the OULAD dataset in a newly created OULAD directory.
2. Next, run JupyterLab with the `jupyter` Makefile target:
   ```bash
   $ make jupyter
   ```
   This should start the JupyterLab server in the project's docker container and display the server's connection URL
   (e.g. http://127.0.0.1:8888/lab?token=52ccae28037a4012e1f4cefc46346f36ba29cea9e935fb14a) to which you can navigate.

### Building with python virtual environment

1. Clone this project and run the `build-venv` Makefile target from the root of the project (where this README.md file is located):
   ```bash
   $ make build-venv
   ```
   This should setup the project and download the OULAD dataset in a newly created OULAD directory.
2. Next, run JupyterLab with the `jupyter-venv` Makefile target:
   ```bash
   $ make jupyter-venv
   ```
   This should start the JupyterLab server and display the server's connection URL
   (e.g. http://127.0.0.1:8888/lab?token=52ccae28037a4012e1f4cefc46346f36ba29cea9e935fb14a) to which you can navigate.
