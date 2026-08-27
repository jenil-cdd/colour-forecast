.PHONY: help install extract panel diagnose backtest rolling recommend report test all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## install python dependencies
	pip install -r requirements.txt

extract:  ## pull raw tables from BigQuery into data/raw (cached; use FORCE=1 to re-query)
	python3 -m src.extract $(if $(FORCE),--force,)

panel:    ## build the modelling panel from data/raw
	python3 -m src.panel

diagnose: ## re-estimate every assumption in the incumbent heuristic
	python3 scripts/diagnose_priors.py

backtest: ## train <= train_end, score the held-out window, all models
	python3 -m src.backtest

rolling:  ## repeat the protocol at earlier launch events
	python3 scripts/run_rolling.py

recommend: ## produce the order sheet
	python3 scripts/run_recommend.py $(or $(MODEL),knn_lookalike) $(or $(SCOPE),duvet_all)

report:   ## assemble reports/FINDINGS.md
	python3 scripts/build_report.py

order-sheet: ## final order sheet for config/candidates.csv
	python3 scripts/run_order_sheet.py $(or $(MODEL),knn_lookalike)

test:     ## run the test suite
	python3 -m pytest tests/ -q

all: extract panel diagnose backtest rolling recommend report order-sheet  ## full pipeline

clean:    ## remove derived data (keeps data/raw)
	rm -rf data/processed/*.parquet data/outputs/* reports/FINDINGS.md
