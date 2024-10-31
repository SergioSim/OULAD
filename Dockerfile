# -- Base image --
FROM python:3.12-slim-bookworm as base

# Upgrade system packages and pip, install security updates and dependencies
RUN apt update && \
    apt -y upgrade && \
    apt -y install gcc git graphviz python3-dev && \
    pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -- Development --
FROM base as development

# Copy requirements.txt
COPY requirements.txt /app/

# Install python dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Un-privileged user running the application
USER ${DOCKER_USER:-1000}

CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
