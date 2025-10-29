.PHONY: setup
setup:
	@echo "creating .venv..."
	@uv sync --locked
	@echo "downloading test models"
	@uv run scripts/download_test_models.py

.PHONY:
format:
	@echo "Formatting python"
	@uv run -m ruff format
	@echo "Formatting rust"
	@cargo fmt
