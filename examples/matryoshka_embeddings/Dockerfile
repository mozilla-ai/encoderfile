FROM encoderfile:latest AS build

RUN apt-get update && apt-get install -y \
    python3 \
    python3.13-venv \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://hf.co/cli/install.sh | bash \
    && mv /root/.local/bin/hf /usr/local/bin/hf

COPY download_model.sh .
COPY encoderfile.yml .
COPY transform.lua .

RUN sh download_model.sh

RUN encoderfile build -f encoderfile.yml

FROM debian:bookworm-slim AS final

COPY --from=build /opt/encoderfile/nomic-embed-text-v1_5.encoderfile /usr/bin/nomic-embed-text-v1_5.encoderfile

ENTRYPOINT ["nomic-embed-text-v1_5.encoderfile"]
