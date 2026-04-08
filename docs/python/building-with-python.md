# Building with Python

The `encoderfile` Python package lets you build encoderfile binaries programmatically — no separate CLI installation required. It is a thin wrapper around the same Rust build pipeline used by the CLI tool.

## Installation

```bash
pip install encoderfile
```

```bash
# or with uv
uv add encoderfile
```

## Prerequisites

You need an ONNX-exported model directory containing:

- `model.onnx` — ONNX model weights
- `tokenizer.json` — tokenizer vocabulary and configuration
- `config.json` — model architecture metadata

Export any HuggingFace model with [Optimum](https://huggingface.co/docs/optimum):

```bash
pip install 'optimum[onnx]'
optimum-cli export onnx \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --task text-classification \
  ./sentiment-model
```

## Quick Start

The simplest build uses `EncoderfileBuilder` directly:

```python
from encoderfile import EncoderfileBuilder, ModelType

builder = EncoderfileBuilder(
    name="sentiment-analyzer",
    model_type=ModelType.SequenceClassification,
    path="./sentiment-model",  # path to your ONNX-exported model directory
)
builder.build()
# writes ./sentiment-analyzer.encoderfile
```

## Three Ways to Build

### 1. `EncoderfileBuilder` (full control)

Best when you need fine-grained control over tokenizer settings, transforms, or cross-compilation targets.

```python
from encoderfile import EncoderfileBuilder, ModelType, TokenizerBuildConfig, Fixed

builder = EncoderfileBuilder(
    name="my-ner-model",
    model_type=ModelType.TokenClassification,
    path="./ner-model",
    output_path="./build/my-ner-model.encoderfile",
    version="1.2.0",
    tokenizer=TokenizerBuildConfig(
        pad_strategy=Fixed(n=512),
        max_length=512,
    ),
)
builder.build()
```

### 2. `build()` convenience function (flat arguments)

Best for scripts where you want to avoid importing supporting classes.

```python
from encoderfile import build, ModelType

build(
    name="my-embedder",
    model_type=ModelType.Embedding,
    path="./embedding-model",
    output_path="./my-embedder.encoderfile",
    tokenizer_pad_to="batch_longest",
    tokenizer_max_length=256,
)
```

### 3. `build_from_config()` (YAML config file)

Best when your build configuration lives in a file alongside your model.

```python
from encoderfile import build_from_config

build_from_config("sentiment-config.yml")
```

Where `sentiment-config.yml` contains:

```yaml
encoderfile:
  name: sentiment-analyzer
  path: ./sentiment-model
  model_type: sequence_classification
  output_path: ./build/sentiment-analyzer.encoderfile
```

## Model Types

See the [Building Guide](../reference/building.md#model-types) for a full description of each model type, including supported HuggingFace `AutoModel` classes and inference output shapes.

`ModelType` values are plain strings (`StrEnum`), so you can pass the string directly instead of importing the enum:

```python
builder = EncoderfileBuilder(
    name="my-model",
    model_type="sequence_classification",
    path="./my-model",
)
```

## Tokenizer Configuration

Override tokenizer padding and truncation settings at build time with `TokenizerBuildConfig`. These settings are baked into the binary and applied at every inference call.

```python
from encoderfile import EncoderfileBuilder, ModelType, TokenizerBuildConfig, BatchLongest, Fixed

# Dynamic padding — each batch is padded to its longest sequence
tokenizer = TokenizerBuildConfig(pad_strategy=BatchLongest())

# Fixed-length padding — every sequence padded/truncated to exactly 512 tokens
tokenizer = TokenizerBuildConfig(
    pad_strategy=Fixed(n=512),
    max_length=512,
    truncation_side="right",
    truncation_strategy="longest_first",
)

builder = EncoderfileBuilder(
    name="my-model",
    model_type=ModelType.Embedding,
    path="./my-model",
    tokenizer=tokenizer,
)
builder.build()
```

When using the `build()` convenience function, use flat `tokenizer_*` arguments instead:

```python
from encoderfile import build, ModelType

build(
    name="my-model",
    model_type=ModelType.Embedding,
    path="./my-model",
    tokenizer_pad_to=512,          # int → Fixed(n=512), or "batch_longest"
    tokenizer_max_length=512,
    tokenizer_truncation_side="right",
)
```

## Lua Transforms

Embed a Lua post-processing script to transform model logits before they are returned. See the [Transforms guide](../transforms/index.md) for the full scripting API.

```python
from encoderfile import EncoderfileBuilder, ModelType

# Inline Lua string
builder = EncoderfileBuilder(
    name="normalized-embedder",
    model_type=ModelType.Embedding,
    path="./embedding-model",
    transform="function Postprocess(logits) return logits:lp_normalize(2.0, 2.0) end",
)
builder.build()
```

```python
# From a file — use the build() convenience function
from encoderfile import build, ModelType

build(
    name="normalized-embedder",
    model_type=ModelType.Embedding,
    path="./embedding-model",
    transform_path="./normalize.lua",
)
```

## Cross-compilation

Build a binary targeting a different platform by passing a `target` triple:

```python
from encoderfile import EncoderfileBuilder, ModelType

builder = EncoderfileBuilder(
    name="my-model",
    model_type=ModelType.Embedding,
    path="./my-model",
    target="x86_64-unknown-linux-gnu",  # build for Linux on a Mac
)
builder.build()
```

You can also use a `TargetSpec` object:

```python
from encoderfile import EncoderfileBuilder, ModelType, TargetSpec

spec = TargetSpec("aarch64-apple-darwin")
print(spec.arch, spec.os, spec.abi)  # "aarch64", "apple", "darwin"

builder = EncoderfileBuilder(
    name="my-model",
    model_type=ModelType.Embedding,
    path="./my-model",
    target=spec,
)
builder.build()
```

## Inspecting a Binary

Use `read_metadata()` to read the metadata embedded in an existing encoderfile binary without running inference:

```python
from encoderfile import read_metadata

info = read_metadata("./sentiment-analyzer.encoderfile")

print(info.encoderfile_config.name)        # "sentiment-analyzer"
print(info.encoderfile_config.model_type)  # "sequence_classification"
print(info.encoderfile_config.version)     # "1.0.0"
print(info.model_config.id2label)          # {0: "NEGATIVE", 1: "POSITIVE"}
```

## Next Steps

- **[Python API Reference](api-reference.md)** — full documentation for every class and function
- **[Transforms Guide](../transforms/index.md)** — custom post-processing with Lua scripts
- **[CLI Reference](../reference/cli.md)** — `build`, `serve`, and `infer` commands for the compiled binary
