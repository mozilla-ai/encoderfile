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
        pkg-config \
        protobuf-compiler \
        libssl-dev && \
        rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ARG RUST_VERSION=1.91.0

RUN rustup toolchain install ${RUST_VERSION}
RUN rustup default ${RUST_VERSION}

FROM base AS build

WORKDIR /app

# copy source
# NOTE: if new modules are added to the Cargo workspaces, they must be added here.
COPY Cargo.toml Cargo.lock ./
COPY encoderfile ./encoderfile
COPY encoderfile-runtime ./encoderfile-runtime

# Build release binary.
RUN cargo build --workspace --release

# ---- Final stage ------------------------------------------------------------
FROM debian:bookworm-slim AS final

# Default working directory.
WORKDIR /opt/encoderfile

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
