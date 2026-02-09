# Serve Matryoshka Sentence Embeddings with `nomic-ai/nomic-embed-text-v1.5`

In this cookbook, we build an Encoderfile that serves Matryoshka sentence embeddings using the `nomic-ai/nomic-embed-text-v1.5` model. You’ll package the model into a single, self-contained binary that runs fully offline and can be deployed as a REST API, gRPC service, or CLI.

Along the way, we show how to apply the model’s recommended Matryoshka post-processing and select a fixed embedding dimensionality at build time, making it easier to balance retrieval quality, latency, and memory footprint in production.

Check out the full code in [Github](https://github.com/mozilla-ai/encoderfile/tree/main/examples/matryoshka_embeddings).

### What are Matryoshka Embeddings?

[Matryoshka embeddings](https://arxiv.org/abs/2205.13147) are embeddings that remain semantically meaningful even when truncated. A single model can produce embeddings at multiple dimensionalities by taking prefixes of the output vector, making it easy to balance retrieval quality against storage and performance constraints in downstream systems.

This Encoderfile is useful when you want to standardize on a fixed embedding size while still benefiting from a Matryoshka-trained model’s training regime. By selecting the embedding dimensionality at build time, you can tailor the binary to your storage, indexing, and memory constraints—then deploy it as a stable, reproducible artifact.

This is a good fit for production search and retrieval systems, offline indexing pipelines, and environments with strict operational or compliance requirements, where embedding shape must be fixed and predictable, and runtime configuration is intentionally limited.

## How Matryoshka Embeddings Are Applied in This Encoderfile

The `nomic-ai/nomic-embed-text-v1.5` model produces token-level hidden states at its full native dimensionality. On their own, these outputs are not directly usable as sentence embeddings. This Encoderfile applies the post-processing steps recommended by the model authors and compiles them directly into the binary.

All post-processing is implemented as a Lua transform and runs inside the Encoderfile at inference time. There is no runtime configuration: the embedding shape and normalization behavior are fixed at build time.


```lua
--8<-- "examples/matryoshka_embeddings/transform.lua"
```

The transform performs the following steps:

1. **Mean pooling**
   Token-level embeddings are averaged across the sequence using the attention mask, producing a single vector per input text.

2. **Layer normalization**
   The pooled embeddings are normalized to stabilize scale and match the model’s reference implementation.

3. **Matryoshka truncation**
   The embedding vector is truncated to a fixed dimensionality (`MatryoshkaDim`). Because the model was trained with a Matryoshka objective, the prefix of the vector remains semantically meaningful even at lower dimensions.

4. **L2 normalization**
   The final embeddings are L2-normalized, making them suitable for cosine similarity and nearest-neighbor search.

By compiling these steps into the Encoderfile, every inference produces embeddings with a fixed, predictable shape and identical semantics across environments. This avoids runtime configuration drift and makes the resulting binary easier to deploy in production systems where embedding dimensionality, memory usage, and indexing behavior must be tightly controlled.

The result is a single, reproducible artifact that serves Matryoshka embeddings at a chosen dimensionality—without requiring downstream systems to understand or reimplement the post-processing logic.

## Building the Encoderfile

## Option 1: Build using Docker

This is the easiest and most reproducible path. All dependencies are pinned and handled for you.

### Step 1: Build the Encoderfile

Run:

```bash
docker build -t nomic-embed-text-v1_5:latest .
```

This step:

- downloads the model artifacts
- applies the Matryoshka post-processing configuration
- builds the final Encoderfile binary

### Step 2: Run the Encoderfile

Run:

```bash
docker run \
    -it \
    -p 8080:8080 \
    -p 50051:50051 \
    nomic-embed-text-v1_5:latest serve
```

The container runs the Encoderfile directly and starts an embedding server. This exposes both an HTTP (port `8080`) and gRPC endpoint (port `50051`). To see more options, run:

```bash
docker run -it nomic-embed-text-v1_5:latest serve --help
```

## Option 2: Build from Scratch

Use this path if you want full control over the build environment or to inspect each step.

### Step 1: Install Prerequisites

Ensure the encoderfile CLI is installed and available in your `PATH`. For instructions on how to install the encoderfile CLI, check out our [Getting Started](https://mozilla-ai.github.io/encoderfile/getting-started/#encoderfile-cli-tool) guide.

To install Huggingface CLI (for downloading model artifacts):

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

### Step 2: Download Model

Run the following:
```
sh download_model.sh
```

This script downloads the `nomic-ai/nomic-embed-text-v1.5` model files expected by the Encoderfile build configuration.

### Step 3: Build the Encoderfile

Run the following:

```bash
encoderfile build -f encoderfile.yml
```

This produces a single executable binary, named `nomic-embed-text-v1_5.encoderfile`. All configuration—model weights, embedding dimensionality, and post-processing logic—is compiled into this file.

### Step 4: Run the Encoderfile

To serve the model as a server:

```bash
# Optional: if you get a permission error
chmod +x ./nomic-embed-text-v1_5.encoderfile

./nomic-embed-text-v1_5.encoderfile serve
```

## Running Inference

You can verify that the server is running by running in a separate terminal:

```bash
curl -X GET \
  -H "Accept: application/json" \
  http://localhost:8080/health
```

You should get back the following:

```text
"OK!"
```

The following Python snippet shows how to extract sentence embeddings:

```python3
import requests

data = {
    "inputs": [
        "this is a sentence",
        "this is another sentence"
        ]
}

response = requests.post(
    "http://localhost:8080/predict",
    json=data
    )

print(response.json())
```
