PY := .venv/bin/python

.PHONY: install test ingest enrich prices build analyze report all clean

install:
	uv venv --python 3.13 .venv
	uv pip install --python $(PY) -e .

test:
	$(PY) -m pytest -q

ingest:
	$(PY) -m prpanel ingest --since 2023-11-07 --max-reviews 50000

enrich:
	$(PY) -m prpanel enrich

prices:
	$(PY) -m prpanel prices

build:
	$(PY) -m prpanel build

analyze:
	$(PY) -m prpanel analyze --n-perm 300

report:
	$(PY) -m prpanel report

all: ingest enrich prices build analyze report

clean:
	rm -rf data/reviews.sqlite* data/analysis data/panel_weekly.parquet data/coverage.csv reports/figures/*.png
