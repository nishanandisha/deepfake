.PHONY: setup lint test build-splits train-visual train-acoustic train-fusion evaluate

setup:
	python -m pip install -r requirements.txt

download-data:
	python scripts/download_data.py --output-dir data/raw/lavdf

inspect-data:
	python scripts/inspect_dataset.py --root data/raw/lavdf --full-index

build-splits:
	python scripts/build_splits.py

warm-cache:
	python scripts/warm_cache.py

lint:
	python -m ruff check src scripts tests

test:
	python -m pytest

train-visual:
	python scripts/train.py model=visual

train-acoustic:
	python scripts/train.py model=acoustic

train-fusion:
	python scripts/train.py model=fusion

evaluate:
	python scripts/eval.py

evaluate-late-fusion:
	python scripts/evaluate_late_fusion.py

calibrate:
	python scripts/calibrate.py

explain:
	python scripts/explain.py
