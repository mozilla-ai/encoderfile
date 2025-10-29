.PHONY: setup
setup:
	@echo "creating .venv..."
	uv sync --locked
	@echo "downloading test models"
	uv run scripts/download_test_models.py
