.PHONY: setup
setup:
	@echo "creating .venv..."
	@uv sync --locked
	@echo "downloading test models..."
	@uv run --group setup scripts/download_test_models.py

.PHONY: format
format:
	@echo "Formatting python..."
	@uv run --dev -m ruff format
	@echo "Formatting rust..."
	@cargo fmt

.PHONY: lint
lint:
	cargo clippy \
		--all-features \
		--all-targets \
		-- \
		-D warnings

.PHONY: coverage
coverage:
	cargo llvm-cov \
		--workspace \
		--all-features \
		--lcov \
		--output-path lcov.info

,PHONY: licenses
licenses:
	@echo "Generating licenses..."
	@cargo about generate about.hbs > THIRDPARTY.md

.PHONY: check
check:
	cargo hack check --each-feature

.PHONY:
clippy:
	@cargo clippy \
		--fix \
		--all-features \
		--allow-dirty

.PHONY: pre-commit
pre-commit:
	@uv run --dev pre-commit run --all-files

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

.PHONY: generate-docs
generate-docs:
# 	generate JSON schema for encoderfile config
	@cargo run \
		--bin generate-encoderfile-config-schema \
		--all-features
