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
build: ## build the docker container
	@$(COMPOSE) build app
.PHONY: build

build-jupyter-book: \
	sync-jupyter-notebooks
build-jupyter-book: ## build the jupyter book
	@bin/jupyter-book build src/jupyterbook
.PHONY: build-jupyter-book

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
	clear-jupyter-book \
	build-jupyter-book
deploy-jupyter-book: ## deploy the jupyter book to gh-pages using the ghp-import package
	ghp-import -n -p -f src/jupyterbook/_build/html
.PHONY: deploy-jupyter-book

down: ## stop and remove the docker container
	@$(COMPOSE) down --rmi all -v --remove-orphans
.PHONY: down

jupyter:  ## run jupyter lab
	@$(COMPOSE_RUN) --publish "${OULAD_JUPYTER_PORT}:${OULAD_JUPYTER_PORT}" app \
	jupyter lab --port "${OULAD_JUPYTER_PORT}" --ip "0.0.0.0" --no-browser \
	--notebook-dir src
.PHONY: jupyter

# Nota bene: Black should come after isort just in case they don't agree...
lint: ## lint back-end python sources
lint: \
  lint-isort \
  lint-black \
  lint-flake8 \
  lint-pylint \
  lint-bandit
.PHONY: lint

lint-black: ## lint python sources with black
	@echo 'lint:black started…'
	@$(COMPOSE_RUN_APP) black src
.PHONY: lint-black

lint-flake8: ## lint python sources with flake8
	@echo 'lint:flake8 started…'
	@$(COMPOSE_RUN_APP) flake8
.PHONY: lint-flake8

lint-isort: ## automatically re-arrange python imports
	@echo 'lint:isort started…'
	@$(COMPOSE_RUN_APP) isort --atomic src
.PHONY: lint-isort

lint-pylint: ## lint python sources with pylint
	@echo 'lint:pylint started…'
	@$(COMPOSE_RUN_APP) pylint src
.PHONY: lint-pylint

lint-bandit: ## lint python sources with bandit
	@echo 'lint:bandit started…'
	@$(COMPOSE_RUN_APP) bandit -qr src
.PHONY: lint-bandit

sync-jupyter-notebooks:  ## synchronize jupyter notebooks
	@bin/jupytext --sync ${JUPYTER_NOTEBOOKS}
.PHONY: sync-jupyter-notebooks

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help
