### For local builds ##########################################################
# IMPORTANT:
# This image intentionally builds on Debian bookworm (glibc 2.36).
# Do NOT switch the build stage to rust:<version> or newer distros,
# or the resulting binary will not run on stable Linux systems.

# ---- Build stage ------------------------------------------------------------
FROM debian:bookworm AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
        jq \
        pkg-config \
        protobuf-compiler \
        python3 \
        python3-venv \
        libssl-dev && \
        rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ARG RUST_VERSION=1.91.0

RUN rustup toolchain install ${RUST_VERSION}
RUN rustup default ${RUST_VERSION}

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install 3.13

# install cargo-binstall
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# install just
RUN cargo binstall just

FROM base AS build

WORKDIR /app

# copy source
# NOTE: if new modules are added to the Cargo workspaces, they must be added here.
COPY Cargo.toml Cargo.lock ./
COPY encoderfile ./encoderfile
COPY encoderfile-runtime ./encoderfile-runtime
COPY encoderfile-py ./encoderfile-py
COPY Justfile README.md LICENSE THIRDPARTY.md ./
COPY pyproject.toml uv.lock ./

# Build release binary.
ARG VARIANT=""
RUN just build "" ${VARIANT}

# ---- Final stage ------------------------------------------------------------
FROM debian:bookworm-slim AS final

# Default working directory.
WORKDIR /opt/encoderfile

#  ca-certificates for downloading base binaries
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install binary into a standard PATH location.
COPY --from=build /app/target/release/encoderfile /usr/local/bin/encoderfile

RUN chmod +x /usr/local/bin/encoderfile

# Add documentation and license material.
RUN mkdir -p /usr/share/docs/encoderfile
COPY README.md THIRDPARTY.md LICENSE /usr/share/docs/encoderfile/

# smoke test
RUN /usr/local/bin/encoderfile version

# Default command entry.
ENTRYPOINT ["/usr/local/bin/encoderfile"]
CMD ["--help"]
