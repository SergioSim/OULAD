# -- Base image --
FROM python:3.10.4-slim-bullseye as base

# Upgrade system packages, install security updates and dependencies
RUN apt update && \
    apt -y upgrade && \
    apt -y install gcc git graphviz python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to its latest release
RUN pip install --upgrade pip

WORKDIR /app

# -- Development --
FROM base as development

# Copy requirements.txt
COPY requirements.txt /app/

# Install python dependencies
RUN pip install -r requirements.txt

# Install jupyter extensions
RUN jupyter nbextension install jupytext --py && \
    jupyter nbextension enable jupytext --py

# Un-privileged user running the application
USER ${DOCKER_USER:-1000}

CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
