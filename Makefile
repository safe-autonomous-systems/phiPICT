SRC_DIR=src/phipict
TEST_DIR=tests
EXAMPLES_DIR=examples

PYTHON ?= python
PYTEST ?= python -m pytest
PIP ?= python -m pip
MAKE ?= make
RUFF ?= ruff
MYPY ?= mypy
PRECOMMIT ?= pre-commit

.PHONY: check-ruff
check-ruff:
	$(RUFF) check ${SRC_DIR} --fix || :
	$(RUFF) check ${TEST_DIR} --fix || :
	$(RUFF) check ${EXAMPLES_DIR} --fix || :

.PHONY: check-mypy
check-mypy:
	$(MYPY) ${SRC_DIR} || :

check: check-ruff check-mypy

.PHONY: stubs
stubs:
	$(PYTHON) -m pybind11_stubgen phipict._C \
		--enum-class-locations "ConvergenceCriterion:phipict._C" \
		-o src
	$(PYTHON) -m pybind11_stubgen phipict._noise -o src


.PHONY: docs
docs:
	$(MAKE) -C docs html SPHINXBUILD="$(PYTHON) -m sphinx"

.PHONY: upload-docs
upload-docs: docs
	# keep the wheel index (whl/) that CI maintains on gh-pages
	git fetch origin gh-pages && git archive origin/gh-pages whl | tar -x -C docs/build/html || :
	ghp-import -n -p -f docs/build/html

.PHONY: pre-commit
pre-commit:
	$(PRECOMMIT) run --all-files || :

.PHONY: format
format:
	$(RUFF) format

.PHONY: test
test:
	$(PYTEST) $(TEST_DIR)

.PHONY: install
install: clean build
	$(PIP) install dist/phipict-*.whl
	$(MAKE) stubs

.PHONY: install-dev
install-dev:
	PHIPICT_BUILD_NOISE_EXT=1 \
	MAX_JOBS=1 \
	TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX" \
	PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
	$(PIP) install --no-build-isolation -e ".[dev]"
	$(MAKE) stubs

.PHONY: clean
clean:
	rm -rf build dist __pycache__ src/*.egg-info
	rm -rf $(SRC_DIR)/**/*.pyc $(SRC_DIR)/**/*.pyo
	rm -rf $(TEST_DIR)/**/*.pyc $(TEST_DIR)/**/*.pyo
	$(MAKE) -C docs clean


.PHONY: build
build: clean
	MAX_JOBS=1 \
	TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX" \
	$(PYTHON) -m pip wheel . -w dist --no-build-isolation --no-deps
