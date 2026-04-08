# Python API Reference

Complete reference for the `encoderfile` Python package.

```python
from encoderfile import (
    EncoderfileBuilder,
    ModelType,
    TokenizerBuildConfig,
    BatchLongest,
    Fixed,
    TargetSpec,
    read_metadata,
    build,
    build_from_config,
)
```

---

## `EncoderfileBuilder`

The primary class for building encoderfile binaries. Validates model files, then embeds ONNX weights, tokenizer configuration, and model metadata into a pre-built base binary before writing the result to disk.

### `EncoderfileBuilder(*, name, model_type, path, ...)`

```python
EncoderfileBuilder(
    *,
    name: str,
    model_type: ModelType | str,
    path: str,
    version: str | None = None,
    output_path: str | None = None,
    cache_dir: str | None = None,
    base_binary_path: str | None = None,
    transform: str | None = None,
    lua_libs: list[str] | None = None,
    tokenizer: TokenizerBuildConfig | None = None,
    validate_transform: bool = True,
    target: str | TargetSpec | None = None,
) -> EncoderfileBuilder
```

All arguments are keyword-only.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Model identifier used in API responses and as the default output filename. |
| `model_type` | `ModelType \| str` | required | Model architecture. Determines how inference outputs are structured. |
| `path` | `str` | required | Path to a directory containing `model.onnx`, `tokenizer.json`, and `config.json`. |
| `version` | `str \| None` | `"0.1.0"` | Model version string embedded in the binary. |
| `output_path` | `str \| None` | `./<name>.encoderfile` | Destination path for the compiled binary. |
| `cache_dir` | `str \| None` | system default | Directory for caching intermediate build artifacts. |
| `base_binary_path` | `str \| None` | `None` | Path to a local pre-built base binary. Skips network download when provided. |
| `transform` | `str \| None` | `None` | Inline Lua post-processing script or file path applied to model logits. |
| `lua_libs` | `list[str] \| None` | `None` | Additional Lua library paths available to the transform script. |
| `tokenizer` | `TokenizerBuildConfig \| None` | `None` | Tokenizer padding and truncation settings. Uses tokenizer defaults when `None`. |
| `validate_transform` | `bool` | `True` | Perform a dry-run validation of the transform script before building. |
| `target` | `str \| TargetSpec \| None` | host platform | Cross-compilation target triple (e.g. `"x86_64-unknown-linux-gnu"`). |

**Example:**

```python
from encoderfile import EncoderfileBuilder, ModelType

builder = EncoderfileBuilder(
    name="sentiment-analyzer",
    model_type=ModelType.SequenceClassification,
    path="./sentiment-model",
    output_path="./build/sentiment-analyzer.encoderfile",
    version="1.0.0",
)
builder.build()
```

---

### `EncoderfileBuilder.from_config(config_path)`

```python
@staticmethod
EncoderfileBuilder.from_config(config_path: str) -> EncoderfileBuilder
```

Create an `EncoderfileBuilder` from a YAML configuration file.

The YAML file must have an `encoderfile` top-level key, containing any of the keywords described in the constructor:

```yaml
encoderfile:
  name: sentiment-analyzer
  version: "1.0.0"
  path: ./models/distilbert-sst2
  model_type: sequence_classification
  output_path: ./build/sentiment-analyzer.encoderfile
```

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `config_path` | `str` | Path to the YAML build configuration file. |

**Raises:** `ValueError` if the config is missing required fields or has invalid values. `FileNotFoundError` if `config_path` does not exist.

---

### `EncoderfileBuilder.build(workdir, version, no_download)`

```python
builder.build(
    workdir: str | None = None,
    version: str | None = None,
    no_download: bool = False,
)
```

Compile and write the encoderfile binary. Validates all model files, runs optional transform validation, embeds assets into the base binary, and writes the output file.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `workdir` | `str \| None` | system temp | Temporary working directory for intermediate build files. |
| `version` | `str \| None` | `None` | Override the encoderfile runtime version to embed. Takes precedence over the version set on the builder. |
| `no_download` | `bool` | `False` | Disable downloading the base binary. Requires `base_binary_path` or a cached binary. |

**Raises:** `FileNotFoundError` if required model files are missing. `ValueError` if the ONNX model is incompatible or the transform fails validation. `RuntimeError` if the binary cannot be written.

---

## `ModelType`

```python
class ModelType(StrEnum):
    Embedding = "embedding"
    SentenceEmbedding = "sentence_embedding"
    SequenceClassification = "sequence_classification"
    TokenClassification = "token_classification"
```

`ModelType` is a `StrEnum` — values are plain strings and can be used interchangeably with their string equivalents.

| Value | String | Use case |
|---|---|---|
| `ModelType.Embedding` | `"embedding"` | Feature extraction, clustering |
| `ModelType.SentenceEmbedding` | `"sentence_embedding"` | Semantic search, similarity |
| `ModelType.SequenceClassification` | `"sequence_classification"` | Sentiment analysis, topic classification |
| `ModelType.TokenClassification` | `"token_classification"` | NER, PII detection |

---

## `TokenizerBuildConfig`

Tokenizer padding and truncation settings baked into the binary at build time. Applied at every inference call.

### `TokenizerBuildConfig(*, pad_strategy, ...)`

```python
TokenizerBuildConfig(
    *,
    pad_strategy: BatchLongest | Fixed | None = None,
    truncation_side: str | None = None,
    truncation_strategy: str | None = None,
    max_length: int | None = None,
    stride: int | None = None,
) -> TokenizerBuildConfig
```

All arguments are keyword-only. Any argument left as `None` uses the value from the model's `tokenizer_config.json`.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `pad_strategy` | `BatchLongest \| Fixed \| None` | tokenizer default | Padding strategy. `BatchLongest()` for dynamic per-batch padding; `Fixed(n=N)` for a fixed sequence length. |
| `truncation_side` | `str \| None` | tokenizer default | Side to truncate from: `"left"` or `"right"`. |
| `truncation_strategy` | `str \| None` | tokenizer default | Truncation algorithm: `"longest_first"`, `"only_first"`, or `"only_second"`. |
| `max_length` | `int \| None` | tokenizer default | Maximum tokens per sequence. Sequences longer than this are truncated. |
| `stride` | `int \| None` | tokenizer default | Token overlap between chunks when splitting long sequences. |

**Example:**

```python
from encoderfile import TokenizerBuildConfig, Fixed

tokenizer = TokenizerBuildConfig(
    pad_strategy=Fixed(n=512),
    max_length=512,
    truncation_side="right",
)
```

---

## `BatchLongest`

```python
class BatchLongest
```

Pad all sequences in a batch to the length of the longest sequence in that batch. Use as `pad_strategy` on `TokenizerBuildConfig`.

```python
from encoderfile import TokenizerBuildConfig, BatchLongest

tokenizer = TokenizerBuildConfig(pad_strategy=BatchLongest())
```

---

## `Fixed`

```python
class Fixed:
    n: int

Fixed(*, n: int) -> Fixed
```

Pad all sequences to a fixed token length `n`. Sequences shorter than `n` are padded; sequences longer than `n` are truncated (subject to `truncation_strategy`).

| Attribute | Type | Description |
|---|---|---|
| `n` | `int` | The fixed sequence length in tokens. |

```python
from encoderfile import TokenizerBuildConfig, Fixed

tokenizer = TokenizerBuildConfig(pad_strategy=Fixed(n=256))
```

---

## `TargetSpec`

```python
class TargetSpec:
    arch: str
    os: str
    abi: str

TargetSpec(spec: str) -> TargetSpec
```

Represents a cross-compilation target platform. Parses a Rust-style target triple string.

| Attribute | Type | Description |
|---|---|---|
| `arch` | `str` | CPU architecture, e.g. `"aarch64"`, `"x86_64"`. |
| `os` | `str` | Operating system, e.g. `"apple"`, `"unknown-linux"`. |
| `abi` | `str` | ABI/environment suffix, e.g. `"darwin"`, `"gnu"`. |

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `spec` | `str` | A Rust-style target triple such as `"aarch64-apple-darwin"` or `"x86_64-unknown-linux-gnu"`. |

```python
from encoderfile import TargetSpec

spec = TargetSpec("aarch64-apple-darwin")
print(spec.arch)  # "aarch64"
print(spec.os)    # "apple"
print(spec.abi)   # "darwin"
```

---

## `read_metadata(path)`

```python
read_metadata(path: str) -> InspectInfo
```

Inspect an encoderfile binary without running inference. Reads the metadata embedded at build time.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `path` | `str` | Filesystem path to a compiled `.encoderfile` binary. |

**Returns:** An `InspectInfo` object.

**Raises:** `FileNotFoundError` if no file exists at `path`. `ValueError` if the file is not a valid encoderfile binary.

```python
from encoderfile import read_metadata

info = read_metadata("./sentiment-analyzer.encoderfile")
print(info.encoderfile_config.name)        # "sentiment-analyzer"
print(info.encoderfile_config.model_type)  # "sequence_classification"
print(info.model_config.id2label)          # {0: "NEGATIVE", 1: "POSITIVE"}
```

---

## `InspectInfo`

Returned by `read_metadata()`.

| Attribute | Type | Description |
|---|---|---|
| `model_config` | `ModelConfig` | Architecture metadata from the embedded `config.json`. |
| `encoderfile_config` | `EncoderfileConfig` | Build-time metadata embedded by `EncoderfileBuilder`. |

---

## `ModelConfig`

Model architecture metadata extracted from the embedded `config.json`.

| Attribute | Type | Description |
|---|---|---|
| `model_type` | `str` | HuggingFace architecture identifier, e.g. `"bert"`, `"distilbert"`. |
| `num_labels` | `int \| None` | Number of output labels for classification models. `None` for embedding models. |
| `id2label` | `dict[int, str] \| None` | Mapping from label index to label string, e.g. `{0: "NEGATIVE", 1: "POSITIVE"}`. |
| `label2id` | `dict[str, int] \| None` | Reverse mapping from label string to index. |

---

## `EncoderfileConfig`

Build-time metadata embedded in the binary.

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Model identifier as specified during the build. |
| `version` | `str` | Model version string, e.g. `"1.0.0"`. |
| `model_type` | `str` | Encoderfile model type string. |
| `transform` | `str \| None` | Inline Lua post-processing script, or `None`. |
| `lua_libs` | `list[str] \| None` | Additional Lua library paths, or `None`. |

---

## Convenience Functions

### `build(**kwargs)`

```python
from encoderfile import build
```

A flat-argument convenience wrapper around `EncoderfileBuilder`. Avoids importing `TokenizerBuildConfig`, `BatchLongest`, and `Fixed` for common use cases. Accepts all the same arguments as `EncoderfileBuilder.__new__` plus `workdir` and `no_download`, with tokenizer settings flattened into `tokenizer_*` prefixed arguments.

**Extra arguments vs `EncoderfileBuilder`:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `transform_str` | `str \| None` | `None` | Inline Lua transform. Mutually exclusive with `transform_path`. |
| `transform_path` | `str \| None` | `None` | Path to a Lua transform file. Mutually exclusive with `transform_str`. |
| `tokenizer_pad_to` | `"batch_longest" \| int \| None` | `None` | Padding strategy: `"batch_longest"` or a fixed length integer. |
| `tokenizer_truncation_side` | `str \| None` | `None` | Truncation side: `"left"` or `"right"`. |
| `tokenizer_truncation_strategy` | `str \| None` | `None` | Truncation strategy: `"longest_first"`, `"only_first"`, `"only_second"`. |
| `tokenizer_max_length` | `int \| None` | `None` | Maximum sequence length in tokens. |
| `tokenizer_stride` | `int \| None` | `None` | Token overlap between sequence chunks. |
| `workdir` | `str \| None` | system temp | Temporary working directory for the build. |
| `no_download` | `bool` | `False` | Disable downloading the base binary. |

```python
from encoderfile import build, ModelType

build(
    name="my-embedder",
    model_type=ModelType.Embedding,
    path="./embedding-model",
    tokenizer_pad_to="batch_longest",
    tokenizer_max_length=256,
)
```

---

### `build_from_config(config_path, workdir, no_download)`

```python
from encoderfile import build_from_config
```

A convenience wrapper around `EncoderfileBuilder.from_config()` that loads a YAML config file and calls `build()` in one step.

```python
build_from_config(
    config_path: str,
    workdir: str | None = None,
    no_download: bool = False,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `config_path` | `str` | required | Path to the YAML build configuration file. |
| `workdir` | `str \| None` | system temp | Temporary working directory for intermediate build files. |
| `no_download` | `bool` | `False` | Disable downloading the base binary. |

```python
from encoderfile import build_from_config

build_from_config("sentiment-config.yml")
```

---

## Enums

### `TokenizerTruncationSide`

```python
class TokenizerTruncationSide(StrEnum):
    Left = "left"
    Right = "right"
```

### `TokenizerTruncationStrategy`

```python
class TokenizerTruncationStrategy(StrEnum):
    LongestFirst = "longest_first"
    OnlyFirst = "only_first"
    OnlySecond = "only_second"
```

These enums are accepted wherever a truncation side or strategy string is expected, but plain strings work equally well.
