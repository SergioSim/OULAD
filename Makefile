# -- Python
PYTHON               = python3

# -- Docker
# Get the current user ID to use for docker run and docker exec commands
DOCKER_UID           = $(shell id -u)
DOCKER_GID           = $(shell id -g)
DOCKER_USER          = $(DOCKER_UID):$(DOCKER_GID)
COMPOSE              = DOCKER_USER=$(DOCKER_USER) docker compose
COMPOSE_RUN          = $(COMPOSE) run --rm
COMPOSE_RUN_APP      = $(COMPOSE_RUN) app

# -- Jupyter
OULAD_JUPYTER_PORT   = 8888
OULAD_DATASET_URL    = https://analyse.kmi.open.ac.uk/open_dataset/download
JUPYTER_NOTEBOOKS    = $(shell find src/notebooks/ -type f -not -path "*.ipynb*")
KEEP_OULAD           = "false"

default: help

# ======================================================================================
# FILES
# ======================================================================================

.jupyter/.jupyter/lab/user-settings/@jupyterlab:
	mkdir -p .jupyter/.jupyter/lab/user-settings/@jupyterlab
	cp -r config/jupyter/* .jupyter/.jupyter/lab/user-settings/@jupyterlab

OULAD:
	mkdir -p OULAD
	wget -O OULAD/dataset.zip $(OULAD_DATASET_URL)
	unzip OULAD/dataset.zip -d OULAD
	rm OULAD/dataset.zip

# ======================================================================================
# RULES
# ======================================================================================

build: \
	.jupyter/.jupyter/lab/user-settings/@jupyterlab \
	OULAD
build: ## build the project environment (with docker)
	@$(COMPOSE) build app
.PHONY: build

build-venv: \
	.jupyter/.jupyter/lab/user-settings/@jupyterlab \
	OULAD
build-venv: ## build the project environment (with venv)
	@$(PYTHON) -m venv venv
	@# Note: make executes each line in a separate shell.
	@# Therefore we chain the subsequent commands.
	@. venv/bin/activate && \
	pip install -U pip setuptools wheel && \
	pip install -r requirements.txt
.PHONY: build-venv

build-jupyter-book: \
	sync-jupyter-notebooks
build-jupyter-book: ## build the jupyter book (with docker)
	@bin/jupyter-book build src/jupyterbook
.PHONY: build-jupyter-book

build-jupyter-book-venv: \
	sync-jupyter-notebooks-venv
build-jupyter-book-venv: ## build the jupyter book (with venv)	
	@. venv/bin/activate && \
	PYTHONPATH=${PWD}/src OULAD_DEFAULT_PATH=${PWD}/OULAD jupyter-book \
		build src/jupyterbook
.PHONY: build-jupyter-book-venv

clear: ## remove temporary project files; to keep the OULAD tables use KEEP_OULAD=true
	@rm -rf .jupyter OULAD src/jupyterbook/notebooks src/jupyterbook/_build .pylint.d
	@if [ ${KEEP_OULAD} = "false" ]; then \
        rm -rf .jupyter OULAD; \
	fi
	@find . -depth -type d -name ".ipynb_checkpoints" -exec rm -rf {} \;
	@find . -depth -type d -name "__pycache__" -exec rm -rf {} \;
.PHONY: clear

clear-jupyter-book: ## remove the jupyter book build files
	@rm -rf src/jupyterbook/_build
.PHONY: clear-jupyter-book

deploy-jupyter-book: \
	build-jupyter-book
deploy-jupyter-book: ## deploy the jupyter book to gh-pages (with docker)
	ghp-import -n -p -f src/jupyterbook/_build/html
.PHONY: deploy-jupyter-book

deploy-jupyter-book-venv: \
	build-jupyter-book-venv
deploy-jupyter-book-venv: ## deploy the jupyter book to gh-pages (with venv)
	@. venv/bin/activate && ghp-import -n -p -f src/jupyterbook/_build/html
.PHONY: deploy-jupyter-book-venv

down: ## stop and remove the docker container
	@$(COMPOSE) down --rmi all -v --remove-orphans
.PHONY: down

down-venv: ## remove the python virual environment
	@rm -rf venv
.PHONY: down-venv

jupyter:  ## run jupyter lab (with docker)
	@$(COMPOSE_RUN) --publish "${OULAD_JUPYTER_PORT}:${OULAD_JUPYTER_PORT}" app \
	jupyter lab --port "${OULAD_JUPYTER_PORT}" --ip "0.0.0.0" --no-browser \
	--notebook-dir src
.PHONY: jupyter

jupyter-venv: ## run jupyter lab (with venv)
	@. venv/bin/activate && \
	PYTHONPATH=${PWD}/src OULAD_DEFAULT_PATH=${PWD}/OULAD jupyter lab \
        --port "${OULAD_JUPYTER_PORT}" \
        --no-browser \
	--ip 127.0.0.1 \
	--notebook-dir src
.PHONY: jupyter-venv

# Nota bene: Black should come after isort just in case they don't agree...
lint: ## lint python sources (with docker)
lint: \
	lint-isort \
	lint-black \
	lint-flake8 \
	lint-pylint \
	lint-bandit
.PHONY: lint

lint-venv: ## lint python sources (with venv)
lint-venv: \
	lint-isort-venv \
	lint-black-venv \
	lint-flake8-venv \
	lint-pylint-venv \
	lint-bandit-venv
.PHONY: lint-venv

lint-black: ## lint python sources with black (with docker)
	@echo 'lint:black started…'
	@$(COMPOSE_RUN_APP) black src
.PHONY: lint-black

lint-black-venv: ## lint python sources with black (with venv)
	@echo 'lint:black started…'
	@. venv/bin/activate && black src
.PHONY: lint-black-venv

lint-flake8: ## lint python sources with flake8 (with docker)
	@echo 'lint:flake8 started…'
	@$(COMPOSE_RUN_APP) flake8
.PHONY: lint-flake8

lint-flake8-venv: ## lint python sources with flake8 (with venv)
	@echo 'lint:flake8 started…'
	@. venv/bin/activate && flake8
.PHONY: lint-flake8-venv

lint-isort: ## automatically re-arrange python imports (with docker)
	@echo 'lint:isort started…'
	@$(COMPOSE_RUN_APP) isort --atomic src
.PHONY: lint-isort

lint-isort-venv: ## automatically re-arrange python imports (with venv)
	@echo 'lint:isort started…'
	@. venv/bin/activate && isort --atomic src
.PHONY: lint-isort-venv

lint-pylint: ## lint python sources with pylint (with docker)
	@echo 'lint:pylint started…'
	@$(COMPOSE_RUN_APP) pylint src
.PHONY: lint-pylint

lint-pylint-venv: ## lint python sources with pylint (with venv)
	@echo 'lint:pylint started…'
	@. venv/bin/activate && pylint src
.PHONY: lint-pylint-venv

lint-bandit: ## lint python sources with bandit (with docker)
	@echo 'lint:bandit started…'
	@$(COMPOSE_RUN_APP) bandit -qr src
.PHONY: lint-bandit

lint-bandit-venv: ## lint python sources with bandit (with venv)
	@echo 'lint:bandit started…'
	@. venv/bin/activate && bandit -qr src
.PHONY: lint-bandit

sync-jupyter-notebooks:  ## synchronize jupyter notebooks (with docker)
	@bin/jupytext --sync ${JUPYTER_NOTEBOOKS}
.PHONY: sync-jupyter-notebooks

sync-jupyter-notebooks-venv:  ## synchronize jupyter notebooks (with venv)
	@. venv/bin/activate && \
	jupytext --sync ${JUPYTER_NOTEBOOKS}
.PHONY: sync-jupyter-notebooks-venv

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help
