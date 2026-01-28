# Encoderfile

![Encoderfile](assets/encoderfile_logo.png)

<p align="center">
  <strong>Deploy Encoder Transformers as self-contained, single-binary executables.</strong>
  <br><br>
  <a href="https://github.com/mozilla-ai/encoderfile">
    <img src="https://img.shields.io/github/v/release/mozilla-ai/encoderfile?style=flat-square" />
  </a>
  <a href="https://github.com/mozilla-ai/encoderfile/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/mozilla-ai/encoderfile?style=flat-square" />
  </a>
</p>

---

**Encoderfile** packages transformer encoders—and their classification heads—into a single, self-contained executable.

Replace fragile, multi-gigabyte Python containers with lean, auditable binaries that have **zero runtime dependencies**[^1]. Written in Rust and built on ONNX Runtime, Encoderfile ensures strict determinism and high performance for financial platforms, content moderation pipelines, and search infrastructure.

## Why Encoderfile?

While **Llamafile** focuses on generative models, **Encoderfile** is purpose-built for encoder architectures. It is designed for environments where compliance, latency, and determinism are non-negotiable.

* **Zero Dependencies:** No Python, no PyTorch, no network calls. Just a fast, portable binary.
* **Smaller Footprint:** Binaries are measured in megabytes, not the gigabytes required for standard container deployments.
* **Protocol Agnostic:** Runs as a REST API, gRPC microservice, CLI tool, or MCP Server out of the box.
* **Compliance-Friendly:** Deterministic and offline-safe, making it ideal for strict security boundaries.

> **Note for Windows users:** Pre-built binaries are not available for Windows. Please see our guide on [building from source](https://mozilla-ai.github.io/encoderfile/latest/reference/building/) for instructions on building from source.

## Use Cases

| Scenario | Application |
|----------|-------------|
| **Microservices** | Run as a standalone gRPC or REST service on localhost or in production. |
| **AI Agents** | Register as an MCP Server to give agents reliable classification tools. |
| **Batch Jobs** | Use the CLI mode (infer) to process text pipelines without spinning up servers. |
| **Edge Deployment** | Deploy sentiment analysis, NER, or embeddings anywhere without Docker or Python. |

## Supported Models

Encoderfile supports encoder-only transformers for:

- **Token Embeddings** - clustering, embeddings (BERT, DistilBERT, RoBERTa)
- **Sequence Classification** - Sentiment analysis, topic classification
- **Token Classification** - Named Entity Recognition, PII detection
- **Sentence Embeddings** - Semantic search, clustering

See our guide on [building from source](https://mozilla-ai.github.io/encoderfile/latest/reference/building/) for detailed instructions on building the CLI tool from source.

Generation models (GPT, T5) are not supported. See [CLI Reference](reference/cli.md) for complete model type details.

## Quick Start

### 1. Install CLI

Download the pre-built CLI tool:

```bash
curl -fsSL https://raw.githubusercontent.com/mozilla-ai/encoderfile/main/install.sh | sh
```

Or build from source (see [Building Guide](reference/building.md)).

### 2. Export Model & Build

Export a HuggingFace model and build it into a binary:

```bash
# Export to ONNX
optimum-cli export onnx --model <model-id> --task <task> ./model

# Build the encoderfile
encoderfile build -f config.yml
```

See the [Building Guide](reference/building.md) for detailed export options and configuration.

### 3. Run & Test

Start the server and make predictions:

```bash
# Start server
./build/sentiment-analyzer.encoderfile serve

# Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Your text here"]}'
```

See the [API Reference](reference/api-reference.md) for complete endpoint documentation.

**Next Steps:** Try the [Token Classification Cookbook](cookbooks/token-classification-ner.md) for a complete walkthrough.

## How It Works

Encoderfile compiles your model into a self-contained binary by embedding ONNX weights, tokenizer, and config directly into Rust code. The result is a portable executable with zero runtime dependencies.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/encoderfile-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/encoderfile-light.svg">
  <img alt="Encoderfile architecture diagram illustrating the build process: compiling ONNX models, tokenizers, and configs into a single binary executable that runs as a zero-dependency gRPC, HTTP, or MCP server." src="assets/encoderfile-light.svg">
</picture>

## Documentation

### Getting Started
- **[Installation & Setup](getting-started.md)** - Complete setup guide from installation to first deployment
- **[Building Guide](reference/building.md)** - Export models and configure builds

### Tutorials
- **[Token Classification (NER)](cookbooks/token-classification-ner.md)** - Build a Named Entity Recognition system
- **[Transforms Guide](transforms/index.md)** - Custom post-processing with Lua scripts

### Reference
- **[CLI Reference](reference/cli.md)** - Full documentation for `build`, `serve`, and `infer` commands
- **[API Reference](reference/api-reference.md)** - REST, gRPC, and MCP endpoint specifications

## Community & Support

- **[GitHub Issues](https://github.com/mozilla-ai/encoderfile/issues)** - Report bugs or request features
- **[Contributing Guide](CONTRIBUTING.md)** - Learn how to contribute
- **[Code of Conduct](CODE_OF_CONDUCT.md)** - Community guidelines

[^1]: Standard builds of Encoderfile require glibc to run because of the ONNX runtime. See [this issue](https://github.com/mozilla-ai/encoderfile/issues/69) on progress on building Encoderfile for musl linux.
