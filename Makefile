.PHONY: help env install install-dev test test-all lint docker docker-test design

PYTHON ?= python3
PIP ?= pip

help:
	@echo "circuit-forge targets:"
	@echo "  make env          - verify Python + ngspice (scripts/check_env.py)"
	@echo "  make install      - pip install runtime deps + editable package"
	@echo "  make install-dev  - same, plus pytest/ruff/mypy"
	@echo "  make test         - fast tests only (no live ngspice end-to-end)"
	@echo "  make test-all     - full suite including slow ngspice tests"
	@echo "  make lint         - ruff + mypy --strict on src/"
	@echo "  make docker       - build the cforge image"
	@echo "  make docker-test  - run check-env and a sample design in Docker"
	@echo "  make design       - cforge design examples/lp_1k.yaml --no-mc"

env:
	$(PYTHON) scripts/check_env.py

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

test:
	$(PYTHON) -m pytest -m "not slow" -q

test-all:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests scripts
	mypy --strict src

docker:
	docker build -t cforge .

docker-test: docker
	docker run --rm cforge check-env
	docker run --rm -v "$(PWD)/out:/work/out" cforge design examples/lp_1k.yaml -o out --no-mc

design:
	cforge design examples/lp_1k.yaml -o out --no-mc
