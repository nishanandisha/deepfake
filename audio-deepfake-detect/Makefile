.PHONY: setup lint test verify predict build-splits warm-cache train evaluate calibrate

setup:
	python -m pip install -r requirements.txt

lint:
	python -m ruff check src scripts tests

test:
	python -m pytest

# Check the shipped artefact rebuilds and runs.
verify:
	python scripts/verify_model.py

# Score the bundled sample clips with the shipped model.
predict:
	python scripts/predict.py samples/real/*.mp4 samples/fake/*.mp4 --no-explain

build-splits:
	python scripts/build_splits.py

warm-cache:
	python scripts/warm_cache.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

calibrate:
	python scripts/calibrate.py --config-name calibration

export:
	python scripts/export_model.py
