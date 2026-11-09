SHELL := /bin/bash
UV := ./.tools/bin/uv
PYTHON := ./.venv/bin/python
TAU2 := ./third_party/tau2-bench
TMP ?= tmp
export PYTHONPYCACHEPREFIX := $(CURDIR)/$(TMP)/pycache
SMOKE_RUN ?= $(TMP)/artifacts/runs/smoke
TRAIN_RUN ?= $(TMP)/artifacts/runs/train
TEACHER_RUN ?= $(TMP)/artifacts/runs/teacher

.PHONY: sync download-model manifest context test profile-check serve-student smoke collect collect-dry normalize teacher normalize-teacher dataset rl-dataset summarize train-sft train-grpo smoke-sft smoke-grpo eval eval-dev compare-eval bundle preflight lint check-shell

sync:
	UV_LINK_MODE=copy $(UV) sync --group dev

download-model:
	$(PYTHON) scripts/download_model.py

manifest:
	$(PYTHON) -m telecom_post_training.cli manifest

context:
	$(PYTHON) -m telecom_post_training.cli context

test:
	$(PYTHON) -m pytest -q

profile-check:
	PROFILE_FILE=configs/profiles/96gb.env bash -c \
		'source scripts/lib_profile.sh; load_training_profile "$$PWD"; exec .venv/bin/python scripts/run_metadata.py validate'

serve-student:
	$(PYTHON) -m telecom_post_training.serve

smoke:
	@echo "Requires a vLLM endpoint and DeepSeek credentials."
	$(PYTHON) -m telecom_post_training.collect --stage smoke --trials 2 --output $(SMOKE_RUN) --execute

collect:
	$(PYTHON) -m telecom_post_training.collect --stage train --trials 4 --output $(TRAIN_RUN) --execute
collect-dry:
	$(PYTHON) -m telecom_post_training.collect --stage train --trials 4 --output $(TRAIN_RUN) --dry-run

normalize:
	$(PYTHON) -m telecom_post_training.cli normalize \
		--input $(TRAIN_RUN) \
		--output $(TMP)/data/processed/train.jsonl \
		--split train

teacher:
	$(PYTHON) -m telecom_post_training.teacher \
		--trials 2 \
		--output $(TEACHER_RUN) \
		--execute

normalize-teacher:
	$(PYTHON) -m telecom_post_training.cli normalize \
		--input $(TEACHER_RUN) \
		--output $(TMP)/data/processed/teacher.jsonl \
		--split teacher

dataset:
	$(PYTHON) -m telecom_post_training.cli build-dataset \
		--input $(TMP)/data/processed/train.jsonl $(TMP)/data/processed/teacher.jsonl \
		--failure-source $(TMP)/data/processed/train.jsonl \
		--output-dir $(TMP)/data/processed/sft \
		--target-tokens 160000

rl-dataset:
	$(PYTHON) -m telecom_post_training.cli build-rl \
		--baseline-input $(TMP)/data/processed/train.jsonl \
		--output-dir $(TMP)/data/processed/rl \
		--task-split train_pool
	$(PYTHON) -m telecom_post_training.cli build-rl \
		--baseline-input $(TMP)/data/processed/train.jsonl \
		--output-dir $(TMP)/data/processed/rl_dev \
		--task-split dev

summarize:
	$(PYTHON) -m telecom_post_training.cli summarize \
		--input $(TMP)/data/processed/train.jsonl \
		--output $(TMP)/artifacts/reports/train_summary.md

train-sft:
	ARM=$${ARM:-uniform} bash scripts/train_sft.sh

train-grpo:
	ARM=$${ARM:-failure_aware} bash scripts/train_grpo.sh

smoke-sft:
	ARM=$${ARM:-uniform} bash scripts/smoke_sft.sh

smoke-grpo:
	ARM=$${ARM:-uniform} bash scripts/smoke_grpo.sh

eval:
	MODEL_LABEL=$${MODEL_LABEL:?Set MODEL_LABEL} bash scripts/evaluate_tau2.sh

eval-dev:
	TASK_SPLIT=dev MODEL_LABEL=$${MODEL_LABEL:?Set MODEL_LABEL} bash scripts/evaluate_tau2.sh

compare-eval:
	$(PYTHON) -m telecom_post_training.cli compare-tasks \
		--left $${BASE_EVAL:-$(TMP)/data/eval/base_all.jsonl} \
		--right $${ARM_EVAL:-$(TMP)/data/eval/failure_aware_all.jsonl} \
		--output $${OUTPUT:-$(TMP)/artifacts/reports/base_vs_failure_aware.md}

bundle:
	$(PYTHON) scripts/build_bundle.py --root $(CURDIR)

preflight:
	bash scripts/preflight.sh

lint:
	./.venv/bin/ruff check src scripts tests

check-shell:
	@for script in scripts/*.sh; do bash -n "$$script" || exit; done
