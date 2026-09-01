.PHONY: format lint fix check

format:
	ruff format .

lint:
	ruff check .

fix:
	ruff check . --fix

check:
	ruff format --check .
	ruff check .
