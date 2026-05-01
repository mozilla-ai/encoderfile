# Setup
setup:
    @echo "creating .venv"
    @uv sync --locked

# Build
dev-py:
    uv run maturin develop \
        -m encoderfile-py/Cargo.toml

build target="" variant="":
    #!/usr/bin/env bash
    set -euo pipefail

    # build encoderfile cli
    just build-encoderfile "{{ target }}"

    # build encoderfile-runtime with EP (if provided)
    if [ -n "{{ variant }}" ]; then
        just build-encoderfile-runtime "{{ target }}" "{{ variant }}"
    fi

    # build encoderfile-runtime (cpu-only)
    just build-encoderfile-runtime "{{ target }}"

    # build python bindings
    just build-encoderfile-py "{{ target }}"

build-encoderfile-py target="":
    #!/usr/bin/env bash
    set -euo pipefail
    target="{{ target }}"
    if [ -z "$target" ]; then
            target=$(rustc -vV | sed -n 's|host: ||p')
        fi

    mkdir -p wheels
    uv run maturin build \
        --release \
        --target $target \
        -m encoderfile-py/Cargo.toml \
        --out wheels \
        -i .venv/bin/python3.13

build-encoderfile target="":
    #!/usr/bin/env bash
    set -euo pipefail
    target="{{ target }}"
    if [ -z "$target" ]; then
            target=$(rustc -vV | sed -n 's|host: ||p')
        fi
    mkdir -p dist/
    version=$(cargo metadata --format-version 1 --no-deps | jq -r '.packages[] | select(.name=="encoderfile-runtime") | .version')

    cargo build \
        --release \
        --target "$target" \
        -p encoderfile

    pkg=encoderfile-$target
    rm -rf "$pkg" && mkdir -p "$pkg"
    cp target/$target/release/encoderfile "$pkg/"

    cp README.md LICENSE THIRDPARTY.md "$pkg/"
    tar -czf "dist/encoderfile-${version}-${target}.tar.gz" -C "$pkg" .
    rm -rf "$pkg"

build-encoderfile-runtime target="" variant="":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p dist/
    target="{{ target }}"
    if [ -z "$target" ]; then
            target=$(rustc -vV | sed -n 's|host: ||p')
        fi
    version=$(cargo metadata --format-version 1 --no-deps | jq -r '.packages[] | select(.name=="encoderfile-runtime") | .version')

    features=$([ -n "{{ variant }}" ] && echo "--features {{ variant }}" || echo "")
    suffix=$([ -n "{{ variant }}" ] && echo "-{{ variant }}" || echo "")

    cargo build \
        --release \
        --target $target \
        -p encoderfile-runtime \
        $features

    pkg=encoderfile-runtime-${target}${suffix}
    rm -rf "$pkg" && mkdir -p "$pkg"
    cp target/$target/release/encoderfile-runtime "$pkg/"

    cp README.md LICENSE THIRDPARTY.md "$pkg/"
    tar -czf "dist/encoderfile-runtime-${version}-${target}${suffix}.tar.gz" -C "$pkg" .
    rm -rf "$pkg"

# Docs
docs: docs-check docs-build

docs-build:
    @uv run python scripts/prepare_gitbook_site.py

docs-check:
    @uv run python scripts/check_docs.py

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

test-py:
    # test encoderfile-py bindings
    @uv run \
        --dev \
        --directory encoderfile-py \
        -m pytest

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
        if [ "${target_size_mb}" -gt "{{ target_max_mb }}" ]; then
            echo "target/ exceeds {{ target_max_mb }} MB — running cargo clean..."
            cargo clean
        else
            echo "target/ size within limits — skipping clean."
        fi
    else
        echo "target/ does not exist — skipping clean."
    fi
