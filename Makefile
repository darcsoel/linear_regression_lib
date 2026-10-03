env:
	uv venv --allow-existing
	uv sync --group dev
	uv run pre-commit install

test:
	uv run pytest -vvv --cov=src tests/

format:
	uv run pre-commit run --all-files
