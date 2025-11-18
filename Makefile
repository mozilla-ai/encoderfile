.PHONY: setup
setup:
# 	@echo "installing dependencies..."
# 	@cargo install cargo-bundle-licenses
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
	@cargo bundle-licenses \
		--format yaml \
		--output THIRDPARTY.yml

.PHONY:
clippy:
	@cargo clippy \
		--fix \
		--all-features \
		--allow-dirty

.PHONY: pre-commit
pre-commit:
	@uv run pre-commit run --all-files

.PHONY: docs
docs:
	@uv run --group docs -m mkdocs serve

# Size threshold in MB
TARGET_MAX_MB ?= 2000

.PHONY: clean
clean:
	@if [ -d target ]; then \
		TARGET_SIZE_MB=$$(du -sm target | cut -f1); \
		echo "target/ size: $${TARGET_SIZE_MB} MB"; \
		if [ "$${TARGET_SIZE_MB}" -gt "$(TARGET_MAX_MB)" ]; then \
			echo "target/ exceeds $(TARGET_MAX_MB) MB — running cargo clean..."; \
			cargo clean; \
		else \
			echo "target/ size within limits — skipping clean."; \
		fi; \
	else \
		echo "target/ does not exist — skipping clean."; \
	fi

.PHONY: generate-schemas
generate-schemas:
	@cargo run \
		--bin generate-encoderfile-config-schema
