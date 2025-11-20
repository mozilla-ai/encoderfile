# Building Encoderfiles from Source

This guide explains how to build custom encoderfile binaries from HuggingFace transformer models.

## Prerequisites

Before building encoderfiles, ensure you have:

- [Rust](https://rust-lang.org/tools/install/) - For building the CLI tool and binaries
- [Python 3.13+](https://www.python.org/downloads/) - For exporting models to ONNX
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package manager
- [protoc](https://protobuf.dev/installation/) - Protocol Buffer compiler

### Installing Prerequisites

**macOS:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install protoc
brew install protobuf
```

**Linux:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install protoc (Ubuntu/Debian)
sudo apt-get install protobuf-compiler

# Install protoc (Fedora)
sudo dnf install protobuf-compiler
```

**Windows:**
```powershell
# Install Rust - Download rustup-init.exe from https://rustup.rs

# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install protoc - Download from https://github.com/protocolbuffers/protobuf/releases
```

## Development Setup

If you're contributing to encoderfile or modifying the source:

```bash
# Clone the repository
git clone https://github.com/mozilla-ai/encoderfile.git
cd encoderfile

# Set up the development environment
make setup
```

This will:
- Install Rust dependencies
- Create a Python virtual environment
- Download model weights for integration tests

## Building the CLI Tool

First, build the encoderfile CLI tool:

```bash
cargo build --bin encoderfile --release
```

The CLI binary will be created at `./target/release/encoderfile`.

Optionally, install it to your system:

```bash
cargo install --path encoderfile --bin encoderfile
```

## Step-by-Step: Building an Encoderfile

### Step 1: Prepare Your Model

You need a HuggingFace model with ONNX weights. You can either export a model or use one with existing ONNX weights.

#### Option A: Export a Model to ONNX

Use `optimum-cli` to export any HuggingFace model:

```bash
optimum-cli export onnx \
  --model <model_id> \
  --task <task_type> \
  <output_directory>
```

**Examples:**

**Embedding model:**
```bash
optimum-cli export onnx \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --task feature-extraction \
  ./models/embedder
```

**Sentiment classifier:**
```bash
optimum-cli export onnx \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --task text-classification \
  ./models/sentiment
```

**NER model:**
```bash
optimum-cli export onnx \
  --model dslim/bert-base-NER \
  --task token-classification \
  ./models/ner
```

**Available task types:**
- `feature-extraction` - For embedding models
- `text-classification` - For sequence classification
- `token-classification` - For token classification (NER, POS tagging, etc.)

See the [HuggingFace task guide](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model) for more options.

#### Option B: Use a Pre-Exported Model

Some models on HuggingFace already have ONNX weights:

```bash
git clone https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english
```

#### Verify Model Structure

Your model directory should contain:

```
my_model/
├── config.json          # Model configuration
├── model.onnx           # ONNX weights (required)
├── tokenizer.json       # Tokenizer (required)
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt
```

### Step 2: Create Configuration File

Create a YAML configuration file (e.g., `config.yml`):

```yaml
encoderfile:
  # Model identifier (used in API responses)
  name: my-model

  # Model version (optional, defaults to "0.1.0")
  version: "1.0.0"

  # Path to model directory
  path: ./models/my-model

  # Model type: embedding, sequence_classification, or token_classification
  model_type: embedding

  # Output path (optional, defaults to ./<name>.encoderfile in current directory)
  output_path: ./build/my-model.encoderfile

  # Cache directory (optional, defaults to system cache)
  cache_dir: ~/.cache/encoderfile

  # Optional: Lua transform for post-processing
  # transform:
  #   path: ./transforms/normalize.lua
```

**Alternative: Specify file paths explicitly:**

```yaml
encoderfile:
  name: my-model
  model_type: embedding
  output_path: ./build/my-model.encoderfile
  path:
    model_config_path: ./models/config.json
    model_weights_path: ./models/model.onnx
    tokenizer_path: ./models/tokenizer.json
```

### Step 3: Build the Encoderfile

Build your encoderfile binary:

```bash
./target/release/encoderfile build -f config.yml
```

Or, if you installed the CLI:

```bash
encoderfile build -f config.yml
```

The build process will:

1. Load and validate the configuration
2. Check for required model files
3. Validate the ONNX model structure
4. Generate a Rust project with embedded assets
5. Compile the project into a self-contained binary
6. Output the binary to the specified path (or `./<name>.encoderfile` if not specified)

**Build output:**
```
./build/my-model.encoderfile
```

### Step 4: Test Your Encoderfile

Make the binary executable and test it:

```bash
chmod +x ./build/my-model.encoderfile

# Test with CLI inference
./build/my-model.encoderfile infer "Test input"

# Or start the server
./build/my-model.encoderfile serve
```

## Configuration Options

### CLI Options

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--config` | `-f` | Yes | Path to YAML configuration file |
| `--output-dir` | - | No | Override output directory from config |
| `--cache-dir` | - | No | Override cache directory from config |
| `--no-build` | - | No | Generate project files without building |

**Examples:**

```bash
# Basic build
./target/release/encoderfile build -f config.yml

# Override output directory
./target/release/encoderfile build -f config.yml --output-dir ./dist

# Generate without building (for debugging)
./target/release/encoderfile build -f config.yml --no-build
```

### Configuration File Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | Yes | - | Model identifier (used in API responses) |
| `path` | Yes | - | Path to model directory or explicit file paths |
| `model_type` | Yes | - | Model type: `embedding`, `sequence_classification`, `token_classification` |
| `version` | No | `"0.1.0"` | Model version string |
| `output_path` | No | `./<name>.encoderfile` | Path where the built binary will be saved |
| `cache_dir` | No | System cache | Where to store generated files |
| `transform` | No | `None` | Optional Lua transform script |
| `build` | No | `true` | Whether to compile the binary |

## Model Types

### Embedding Models

For models using `AutoModel` or `AutoModelForMaskedLM`:

```yaml
encoderfile:
  name: my-embedder
  path: ./models/embedding-model
  model_type: embedding
  output_path: ./build/my-embedder.encoderfile
```

**Examples:**
- `bert-base-uncased`
- `distilbert-base-uncased`
- `sentence-transformers/all-MiniLM-L6-v2`

### Sequence Classification Models

For models using `AutoModelForSequenceClassification`:

```yaml
encoderfile:
  name: my-classifier
  path: ./models/classifier-model
  model_type: sequence_classification
  output_path: ./build/my-classifier.encoderfile
```

**Examples:**
- `distilbert-base-uncased-finetuned-sst-2-english` (sentiment)
- `roberta-large-mnli` (natural language inference)
- `facebook/bart-large-mnli` (entailment)

### Token Classification Models

For models using `AutoModelForTokenClassification`:

```yaml
encoderfile:
  name: my-ner
  path: ./models/ner-model
  model_type: token_classification
  output_path: ./build/my-ner.encoderfile
```

**Examples:**
- `dslim/bert-base-NER`
- `bert-base-cased-finetuned-conll03-english`
- `dbmdz/bert-large-cased-finetuned-conll03-english`

## Advanced Features

### Lua Transforms

Add custom post-processing with Lua scripts:

```yaml
encoderfile:
  name: my-model
  path: ./models/my-model
  model_type: token_classification
  transform:
    path: ./transforms/softmax_logits.lua
```

**Inline transform:**
```yaml
encoderfile:
  name: my-model
  path: ./models/my-model
  model_type: embedding
  transform: "return normalize(output)"
```

### Custom Cache Directory

Specify a custom cache location:

```yaml
encoderfile:
  name: my-model
  path: ./models/my-model
  model_type: embedding
  cache_dir: /tmp/encoderfile-cache
```

## Troubleshooting

### Error: "No such file: model.onnx"

**Solution:** Ensure your model directory contains ONNX weights.

```bash
# Export with optimum-cli
optimum-cli export onnx --model <model_id> --task <task> <output_dir>
```

### Error: "Could not locate model config at path"

**Solution:** The model directory is missing required files (config.json, tokenizer.json, model.onnx).

```bash
# Check directory contents
ls -la ./path/to/model
```

### Error: "cargo build failed"

**Solution:** Check that Rust and dependencies are installed.

```bash
rustc --version
cargo --version
protoc --version
```

### Build is very slow

**Solution:** The first build compiles many dependencies. Subsequent builds will be faster. Use release mode for production:

```bash
# Debug builds are slow
cargo build --bin encoderfile

# Release builds are optimized
cargo build --bin encoderfile --release
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Encoderfile

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install protoc
        run: sudo apt-get install -y protobuf-compiler

      - name: Export model to ONNX
        run: |
          pip install optimum[exporters]
          optimum-cli export onnx \
            --model distilbert-base-uncased \
            --task feature-extraction \
            ./model

      - name: Create config
        run: |
          cat > config.yml <<EOF
          encoderfile:
            name: my-model
            path: ./model
            model_type: embedding
            output_path: ./build/my-model.encoderfile
          EOF

      - name: Build encoderfile
        run: |
          cargo build --bin encoderfile --release
          ./target/release/encoderfile build -f config.yml

      - uses: actions/upload-artifact@v3
        with:
          name: encoderfile
          path: ./build/*.encoderfile
```

## Binary Distribution

After building, your encoderfile binary is completely self-contained:

- No Python runtime required
- No external dependencies
- No network calls needed
- Portable across systems with the same architecture

You can distribute the binary by:

1. Copying it to the target system
2. Making it executable: `chmod +x my-model.encoderfile`
3. Running it: `./my-model.encoderfile serve`

## Next Steps

- [CLI Reference](https://mozilla-ai.github.io/encoderfile/cli/) - Complete command-line documentation
- [API Reference](https://mozilla-ai.github.io/encoderfile/api-reference/) - REST, gRPC, and MCP APIs
- [Getting Started Guide](https://mozilla-ai.github.io/encoderfile/getting-started/) - Step-by-step tutorial
- [Contributing](CONTRIBUTING.md) - Help improve encoderfile
