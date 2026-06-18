# FPS-Voronoi — common commands.
#
# Targets call the repo venv's interpreter directly, so no `source activate`
# is needed. Data flags are variables — override on the command line, e.g.:
#
#     make phase1-real DATASET=kitti FRAME=10
#     make temporal NUM_FRAMES=70 MAX_RANGE=30
#
# Run `make help` (or just `make`) for the full list.

PY        := venv/bin/python
PIP       := venv/bin/pip

# ── Real-data flags (override as needed) ─────────────────────────────────────
DATASET    ?= nuscenes
FRAME      ?= 0
DIMS       ?= 3
MAX_RANGE  ?= 40
MIN_Z      ?= -1.5
NUM_FRAMES ?= 30
START      ?= 0

REAL_FLAGS := --dataset $(DATASET) --dims $(DIMS) --max-range $(MAX_RANGE) --min-z $(MIN_Z)

.DEFAULT_GOAL := help

# ── Setup ────────────────────────────────────────────────────────────────────
.PHONY: venv
venv: ## Create the virtual environment (if missing)
	test -d venv || python -m venv venv

.PHONY: setup
setup: venv ## Create venv and install requirements
	$(PIP) install -r requirements.txt

# ── Phase 1 — primitives + visualization ─────────────────────────────────────
.PHONY: phase1
phase1: ## Phase 1 demo on the synthetic 2-D scene
	$(PY) phase_1/demo.py

.PHONY: phase1-real
phase1-real: ## Phase 1 demo on a real frame (DATASET/FRAME/DIMS/MAX_RANGE/MIN_Z)
	$(PY) phase_1/demo.py $(REAL_FLAGS) --frame $(FRAME)

# ── Phase 2 — fused pass + detectors ─────────────────────────────────────────
.PHONY: phase2
phase2: ## Phase 2 demo on the synthetic 2-D scene
	$(PY) phase_2/demo.py

.PHONY: phase2-real
phase2-real: ## Phase 2 demo on a real frame (DATASET/FRAME/DIMS/MAX_RANGE/MIN_Z)
	$(PY) phase_2/demo.py $(REAL_FLAGS) --frame $(FRAME)

# ── Phase 3 — temporal loop ──────────────────────────────────────────────────
.PHONY: temporal
temporal: ## Frame-to-frame stability + chamfer baseline (NUM_FRAMES/START/...)
	$(PY) phase_3/temporal_demo.py $(REAL_FLAGS) \
		--num-frames $(NUM_FRAMES) --start $(START) --baseline

# ── Tests (run per-phase: a single pytest over all three collides on conftest)
.PHONY: test-phase1 test-phase2 test-phase3
test-phase1: ## Run Phase 1 tests
	$(PY) -m pytest phase_1/tests/ -q

test-phase2: ## Run Phase 2 tests
	$(PY) -m pytest phase_2/tests/ -q

test-phase3: ## Run Phase 3 tests
	$(PY) -m pytest phase_3/tests/ -q

.PHONY: test
test: test-phase1 test-phase2 test-phase3 ## Run every phase's tests in sequence

# ── Housekeeping ─────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove generated demo plots and __pycache__
	rm -f temporal_demo.png
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
