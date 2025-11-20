FROM rust:1.91 AS build

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY encoderfile ./encoderfile
COPY encoderfile-core ./encoderfile-core

ENV ENCODERFILE_DEV=false

RUN cargo build --bin encoderfile --release

FROM rust:1.91 AS final

WORKDIR /data

RUN apt-get update && \
    apt-get install \
        -y --no-install-recommends \
        protobuf-compiler \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /app/target/release/encoderfile /usr/local/bin/encoderfile

ENTRYPOINT ["/usr/local/bin/encoderfile"]

CMD ["--help"]
