### For local builds ##########################################################

# ---- Build stage ------------------------------------------------------------
# Full Rust image to ensure C/C++ toolchains and proto deps are available.
FROM rust:1.91 AS build

WORKDIR /app

# copy source
# NOTE: if new modules are added to the Cargo workspaces, they must be added here.
COPY Cargo.toml Cargo.lock ./
COPY encoderfile ./encoderfile
COPY encoderfile-core ./encoderfile-core
COPY encoderfile-utils ./encoderfile-utils

# Build flag used by the application.
ENV ENCODERFILE_DEV=false

# Build release binary.
RUN cargo build --bin encoderfile --release


# ---- Final stage ------------------------------------------------------------
# Final image must include the full Rust toolchain, since encoderfile
# generates Rust code and performs on-the-fly cargo builds.
FROM rust:1.91 AS final

# Default working directory.
WORKDIR /opt/encoderfile

# Install runtime dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        protobuf-compiler \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install binary into a standard PATH location.
COPY --from=build /app/target/release/encoderfile /usr/local/bin/encoderfile

RUN chmod +x /usr/local/bin/encoderfile

# Add documentation and license material.
RUN mkdir -p /usr/share/docs/encoderfile
COPY README.md THIRDPARTY.md LICENSE /usr/share/docs/encoderfile/

# Default command entry.
ENTRYPOINT ["/usr/local/bin/encoderfile"]
CMD ["--help"]
