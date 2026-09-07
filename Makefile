.PHONY: format lint fix check deploy deploy-ocr deploy-gemma deploy-fastapi modal-stop

format:
	ruff format .

lint:
	ruff check .

fix:
	ruff check . --fix

check:
	ruff format --check .
	ruff check .

# Deploys run in order: the GPU services first, then the FastAPI app that calls
# them. .NOTPARALLEL keeps that order even under `make -j`.
.NOTPARALLEL:

deploy: deploy-ocr deploy-gemma deploy-fastapi

deploy-ocr:
	uv run modal deploy src/ocr/modal_app.py

deploy-gemma:
	uv run modal deploy src/extract/modal_app.py

deploy-fastapi:
	uv run modal deploy fastapi_deploy.py

# Stops every app that is still running (deployed or ephemeral).
modal-stop:
	@uv run modal app list --json \
		| uv run python -c "import json, sys; [print(a['app_id']) for a in json.load(sys.stdin) if a['state'] in ('deployed', 'ephemeral')]" \
		| while read -r app_id; do \
			echo "Stopping $$app_id"; \
			uv run modal app stop "$$app_id" --yes; \
		done
