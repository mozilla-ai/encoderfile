FROM python:3.13-slim AS download_model

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://hf.co/cli/install.sh | bash \
    && mv /root/.local/bin/hf /usr/local/bin/hf

COPY download_model.sh .

# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}

# NOTE: Hugging Face may rate-limit unauthenticated downloads.
# If this build fails with 429 errors, uncomment the lines above and pass an access token
# (not recommended for prod/anywhere where security actually matters):
#
#   docker build --build-arg HF_TOKEN=your_token_here ...
#
# Or download the model assets ahead of time, mount them as a volume, and COPY
# instead of fetching them during the Docker build.
RUN sh download_model.sh

FROM ghcr.io/mozilla-ai/encoderfile:0.3.0-rc.1 AS build

COPY --from=download_model /app/model /opt/encoderfile/model

COPY encoderfile.yml .
COPY transform.lua .

RUN encoderfile build -f encoderfile.yml

FROM debian:bookworm-slim AS final

COPY --from=build \
    /opt/encoderfile/nomic-embed-text-v1_5.encoderfile \
    /usr/bin/nomic-embed-text-v1_5.encoderfile

ENTRYPOINT ["nomic-embed-text-v1_5.encoderfile"]
