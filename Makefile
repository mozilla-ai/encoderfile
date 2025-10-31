.PHONY: setup
setup:
	@echo "installing dependencies..."
	@cargo install cargo-bundle-licenses
	@echo "creating .venv..."
	@uv sync --locked
	@echo "downloading test models..."
	@uv run -m encoderbuild.utils.download_test_models

.PHONY: format
format:
	@echo "Formatting python..."
	@uv run -m ruff format
	@echo "Formatting rust..."
	@cargo fmt

,PHONY: licenses
licenses:
	@echo "Generating licenses..."
	@cargo bundle-licenses --format yaml --output THIRDPARTY.yml
