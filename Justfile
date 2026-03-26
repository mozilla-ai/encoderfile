# Setup
setup:
    @echo "creating .venv"
    @uv sync --locked

# Build
build-py:
    uv run maturin develop \
        -m encoderfile-py/Cargo.toml

# Docs
docs:
    @uv run --group docs -m mkdocs serve

# Check, Test, & Coverage

check:
    @cargo hack check --each-feature

coverage:
	@cargo llvm-cov \
		--workspace \
		--all-features \
		--lcov \
		--output-path lcov.info

pre-commit:
    @uv run --dev pre-commit run --all-files

# Lint & Format
[parallel]
format: format-rs format-py

format-rs:
    @cargo fmt

format-py:
    @uv run --dev -m ruff format

[parallel]
lint: lint-rs lint-py

lint-rs:
    @cargo clippy \
        --all-features \
        --all-targets \
        -- \
        -D warnings

lint-py:
    # ruff check
    @uv run ruff check

stubtest:
    @uv run \
        --dev \
        stubtest \
        --allowlist=allowlist.txt \
        encoderfile._core

# Assets
generate-test-models:
    @uv run \
        --group models \
        scripts/generate_dummy_models.py

generate-docs:
    # JSON schema for encoderfile config
    @cargo run \
        --bin generate-encoderfile-config-schema \
        --all-features

licenses:
	@echo "Generating licenses..."
	@cargo about generate about.hbs > THIRDPARTY.md

# Utilities
target_max_mb := "2000"

clean:
    #!/usr/bin/env bash
    if [ -d target ]; then
        target_size_mb=$(du -sm target | cut -f1)
        echo "target/ size: ${target_size_mb} MB"
        if [ "${target_size_mb}" -gt "{{target_max_mb}}" ]; then
            echo "target/ exceeds {{target_max_mb}} MB — running cargo clean..."
            cargo clean
        else
            echo "target/ size within limits — skipping clean."
        fi
    else
        echo "target/ does not exist — skipping clean."
    fi
