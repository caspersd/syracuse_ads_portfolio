#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = political_analysis
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS 						                                               #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements: .env_updated

.env_created: environment.yml
	# Set channel priority to strict
	conda config --set channel_priority strict
	# Update the environment based on the .yml file
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	@echo "Run the following command to activate your environment:"
	@echo "conda activate $(PROJECT_NAME)"
	touch .env_updated


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8  vopinionminer
	isort --check --diff --profile black opinionminer
	black --check --config ../opinion_extraction/pyproject.toml ../opinion_extraction/opinionminer

## Format source code with black
.PHONY: format
format:
	black --config ../opinion_extraction/pyproject.toml ../opinion_extraction/opinionminer


## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
