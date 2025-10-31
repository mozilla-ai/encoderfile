# Encoderfile

## 🚀 Overview

Encoderfile packages transformer encoders—optionally with classification heads—into a single, self-contained executable.
No Python runtime, no dependencies, no network calls. Just a fast, portable binary that runs anywhere.

While Llamafile focuses on generative models, Encoderfile is purpose-built for encoder architectures with optional classification heads. It supports embedding, sequence classification, and token classification models—covering most encoder-based NLP tasks, from text similarity to classification and tagging—all within one compact binary.

Under the hood, Encoderfile uses ONNX Runtime for inference, ensuring compatibility with a wide range of transformer architectures.

**Why?**

- **Smaller footprint:** a single binary measured in tens-to-hundreds of megabytes, not gigabytes of runtime and packages
- **Compliance-friendly:** deterministic, offline, security-boundary-safe
- **Integration-ready:** drop into existing systems as a CLI, microservice, or API without refactoring your stack

Encoderfiles can run as:
- REST API
- gRPC microservice
- CLI
- (Future) MCP server
- (Future) FFI support for near-universal cross-language embedding

## 🧰 Setup

Prerequisites:
- Rust
- Python
- uv
- protoc

To set up your dev environment, run the following:
```sh
make setup
```

This will install Rust dependencies, create a virtual environment, and download model weights for integration tests (these will show up in `models/`).

If you are using VSCode with the `rust-analyzer` plugin, it will want to automatically compile for you. If the errors become annoying, you can generate a default .env file for the embedding model used in unit tests:
```sh
uv run -m encoderbuild.utils.create_dummy_env_file > .env
```

## 🏗️ Building an Encoderfile

To create an Encoderfile, you must have a HuggingFace model downloaded in an accessible directory. The model directory **must** have exported ONNX weights. It should look like this:

```
my_model/
├── config.json
├── model.onnx
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt
```

Once you have this, run the following command:
```sh
uv run -m encoderbuild build \
    -n my-model-name \
    -t [embedding|sequence_classification|token_classification] \
    -m path/to/model/dir
```

Your final binary is `target/release/encoderfile`. To run it as a server:

```
chmod +x target/release/encoderfile
./target/release/encoderfile serve
```
