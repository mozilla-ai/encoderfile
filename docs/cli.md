# Encoderfile CLI Documentation

## Overview

Encoderfile provides two command-line tools:

1. **`cli`** - Rust-based build tool for creating encoderfile binaries from ONNX models
2. **`encoderfile`** - Rust-based runtime binary for serving models and running inference

## Build Tool: `cli`

The `cli` build command compiles HuggingFace transformer models (with ONNX weights) into self-contained executable binaries using a YAML configuration file.

### `build`

Validates a model configuration and builds a self-contained Rust binary with embedded model assets.

#### Usage

```bash
# If you haven't installed the CLI tool yet, build it first:
cargo build -p encoderfile --bin cli --release

# Then run it:
./target/release/cli build -f <config.yml> [OPTIONS]

# Or install it to your system:
cargo install --path encoderfile --bin cli
cli build -f <config.yml> [OPTIONS]
```

#### Options

| Option | Short | Type | Required | Description |
|--------|-------|------|----------|-------------|
| `--config` | `-f` | Path | Yes | Path to YAML configuration file |
| `--output-dir` | - | Path | No | Override output directory from config |
| `--cache-dir` | - | Path | No | Override cache directory from config |
| `--no-build` | - | Flag | No | Generate project files without building |

#### Configuration File Format

Create a YAML configuration file (e.g., `config.yml`) with the following structure:

```yaml
encoderfile:
  # Model identifier (used in API responses)
  name: my-model

  # Model version (optional, defaults to "0.1.0")
  version: "1.0.0"

  # Path to model directory or explicit file paths
  path: ./models/my-model
  # OR specify files explicitly:
  # path:
  #   model_config_path: ./models/config.json
  #   model_weights_path: ./models/model.onnx
  #   tokenizer_path: ./models/tokenizer.json

  # Model type: embedding, sequence_classification, or token_classification
  model_type: embedding

  # Output directory (optional, defaults to current directory)
  output_dir: ./build

  # Cache directory (optional, defaults to system cache)
  cache_dir: ~/.cache/encoderfile

  # Optional transform (Lua script for post-processing)
  transform:
    path: ./transforms/normalize.lua
  # OR inline transform:
  # transform: "return normalize(output)"

  # Whether to build the binary (optional, defaults to true)
  build: true
```

#### Model Types

- **`embedding`** - For models using `AutoModel` or `AutoModelForMaskedLM`
  - Outputs: `last_hidden_state` with shape `[batch_size, sequence_length, hidden_size]`
- **`sequence_classification`** - For models using `AutoModelForSequenceClassification`
  - Outputs: `logits` with shape `[batch_size, num_labels]`
- **`token_classification`** - For models using `AutoModelForTokenClassification`
  - Outputs: `logits` with shape `[batch_size, num_tokens, num_labels]`

#### Examples

**Build an embedding model:**

Create `embedding-config.yml`:
```yaml
encoderfile:
  name: sentence-embedder
  version: "1.0.0"
  path: ./models/all-MiniLM-L6-v2
  model_type: embedding
  output_dir: ./build
```

Build:
```bash
./target/release/cli build -f embedding-config.yml
```

**Build a sentiment classifier:**

Create `sentiment-config.yml`:
```yaml
encoderfile:
  name: sentiment-analyzer
  path: ./models/distilbert-sst2
  model_type: sequence_classification
```

Build:
```bash
./target/release/cli build -f sentiment-config.yml
```

**Build a NER model with transform:**

Create `ner-config.yml`:
```yaml
encoderfile:
  name: ner-tagger
  path: ./models/bert-ner
  model_type: token_classification
  transform:
    path: ./transforms/softmax_logits.lua
```

Build:
```bash
./target/release/cli build -f ner-config.yml
```

**Generate without building:**
```bash
./target/release/cli build -f config.yml --no-build
```

**Override output directory:**
```bash
./target/release/cli build -f config.yml --output-dir ./custom-output
```

#### Build Process

The `build` command performs the following steps:

1. **Loads configuration** - Parses the YAML config file
2. **Validates model files** - Checks for required files:
   - `model.onnx` - ONNX model weights (or path specified in config)
   - `tokenizer.json` - Tokenizer configuration (or path specified in config)
   - `config.json` - Model configuration (or path specified in config)
3. **Validates ONNX model** - Checks the ONNX model structure and compatibility
4. **Generates project** - Creates a new Rust project in the cache directory with:
   - `main.rs` - Generated from Tera templates
   - `Cargo.toml` - Generated with proper dependencies
5. **Embeds assets** - Uses the `factory!` macro to embed model files at compile time
6. **Compiles binary** - Runs `cargo build --release` on the generated project
7. **Outputs binary** - Copies the binary to `<output_dir>/<name>.encoderfile`

#### Output

Upon successful build, you'll find the binary at:
```
<output_dir>/<name>.encoderfile
```

For example, with `name: my-model` and `output_dir: ./build`:
```
./build/my-model.encoderfile
```

This binary is completely self-contained and includes:
- ONNX model weights (embedded at compile time)
- Tokenizer configuration (embedded)
- Model metadata (embedded)
- Full inference runtime

#### Requirements

Before building, ensure you have:

- [Rust](https://rustup.rs/) toolchain
- [protoc](https://protobuf.dev/) Protocol Buffer compiler
- Valid ONNX model files

#### Troubleshooting

**Error: "No such file: model.onnx"**
```
Solution: Ensure your model directory contains ONNX weights.
Export with: optimum-cli export onnx --model <model_id> --task <task> <output_dir>
```

**Error: "Could not locate model config at path"**
```
Solution: The model directory is missing required files.
Ensure the directory contains: config.json, tokenizer.json, and model.onnx
```

**Error: "No such directory"**
```
Solution: The path specified in the config file doesn't exist.
Check the path value in your YAML config.
```

**Error: "cargo build failed"**
```
Solution: Check that Rust and required system dependencies are installed.
Run: rustc --version && cargo --version
```

**Error: "Cannot locate cache directory"**
```
Solution: System cannot determine the cache directory.
Specify an explicit cache_dir in your config file.
```

---

### `version`

Prints the encoderfile version.

#### Usage

```bash
./target/release/cli version
```

#### Output

```
Encoderfile 0.1.0
```

---

## Runtime Binary: `encoderfile`

After building with the `cli` tool, the resulting `.encoderfile` binary provides inference capabilities.

### Architecture

The runtime CLI is built with the following components:
- **Server Mode**: Hosts models via HTTP and/or gRPC endpoints
- **Inference Mode**: Performs one-off inference operations from the command line
- **Multi-Model Support**: Automatically detects and routes to the appropriate model type

### Commands

### `serve`

Starts the encoderfile server with HTTP and/or gRPC endpoints for model inference.

#### Usage

```bash
encoderfile serve [OPTIONS]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--grpc-hostname` | String | `[::]` | Hostname/IP address for the gRPC server |
| `--grpc-port` | String | `50051` | Port for the gRPC server |
| `--http-hostname` | String | `0.0.0.0` | Hostname/IP address for the HTTP server |
| `--http-port` | String | `8080` | Port for the HTTP server |
| `--disable-grpc` | Boolean | `false` | Disable the gRPC server |
| `--disable-http` | Boolean | `false` | Disable the HTTP server |

#### Examples

**Start both HTTP and gRPC servers (default):**
```bash
encoderfile serve
```

**Start only HTTP server:**
```bash
encoderfile serve --disable-grpc
```

**Start only gRPC server:**
```bash
encoderfile serve --disable-http
```

**Custom ports and hostnames:**
```bash
encoderfile serve \
  --http-hostname 127.0.0.1 \
  --http-port 3000 \
  --grpc-hostname localhost \
  --grpc-port 50052
```

#### Notes

- At least one server type (HTTP or gRPC) must be enabled
- The server will display a banner upon successful startup
- Both servers run concurrently using async tasks

---

### `infer`

Performs inference on input text using the configured model. The model type is automatically detected based on configuration.

#### Usage

```bash
encoderfile infer <INPUTS>... [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<INPUTS>` | Yes | One or more text strings to process |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-f, --format` | Enum | `json` | Output format (currently only JSON supported) |
| `-o, --out-dir` | String | None | Output file path; if not provided, prints to stdout |

#### Model Types

The inference behavior depends on the model type configured:

##### 1. Embedding Models
Generates vector embeddings for input text.

**Example:**
```bash
encoderfile infer "Hello world" "Another sentence"
```

**With normalization disabled:**
```bash
encoderfile infer "Hello world" --normalize=false
```

##### 2. Sequence Classification Models
Classifies entire sequences (e.g., sentiment analysis, topic classification).

**Example:**
```bash
encoderfile infer "This product is amazing!" "I'm very disappointed"
```

##### 3. Token Classification Models
Labels individual tokens (e.g., Named Entity Recognition, Part-of-Speech tagging).

**Example:**
```bash
encoderfile infer "Apple Inc. is located in Cupertino, California"
```

#### Output Formats

Currently, only JSON format is supported (`--format json`). The output structure varies by model type:

##### Embedding Output
```json
{
  "embeddings": [
    [0.123, -0.456, 0.789, ...],
    [0.321, -0.654, 0.987, ...]
  ],
  "metadata": null
}
```

##### Sequence Classification Output
```json
{
  "predictions": [
    {
      "label": "POSITIVE",
      "score": 0.9876
    },
    {
      "label": "NEGATIVE",
      "score": 0.8765
    }
  ],
  "metadata": null
}
```

##### Token Classification Output
```json
{
  "predictions": [
    {
      "tokens": ["Apple", "Inc.", "is", "located", "in", "Cupertino", ",", "California"],
      "labels": ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC", "O", "B-LOC"]
    }
  ],
  "metadata": null
}
```

#### Saving Output to File

**Save results to a file:**
```bash
encoderfile infer "Sample text" -o results.json
```

**Process multiple inputs and save:**
```bash
encoderfile infer "First input" "Second input" "Third input" --out-dir output.json
```

## Configuration

The CLI relies on external configuration to determine:
- Model type (Embedding, SequenceClassification, TokenClassification)
- Model path and parameters
- Server settings

Ensure your configuration is properly set before running commands. Refer to the main encoderfile configuration documentation for details.

## Error Handling

The CLI will return appropriate error messages for:
- Invalid configuration (e.g., both servers disabled)
- Missing required arguments
- Model loading failures
- Inference errors
- File I/O errors

## Examples

### Basic Inference Workflow

```bash
# Set up configuration (example)
export MODEL_PATH=/path/to/model
export MODEL_TYPE=embedding

# Run inference
encoderfile infer "Hello world"

# Save to file
encoderfile infer "Hello world" -o embedding.json
```

### Server Workflow

```bash
# Terminal 1: Start server
encoderfile serve --http-port 8080

# Terminal 2: Make HTTP requests (using curl)
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello world"], "normalize": true}'
```

### Batch Processing

```bash
# Process multiple inputs at once
encoderfile infer \
  "First document to analyze" \
  "Second document to analyze" \
  "Third document to analyze" \
  --out-dir batch_results.json
```

### Custom Server Configuration

```bash
# Run on specific network interface with custom ports
encoderfile serve \
  --http-hostname 192.168.1.100 \
  --http-port 3000 \
  --grpc-hostname 192.168.1.100 \
  --grpc-port 50052
```

## Troubleshooting

### Both servers cannot be disabled
**Error**: "Cannot disable both gRPC and HTTP"

**Solution**: Enable at least one server type:  
```bash
encoderfile serve --disable-grpc  # Keep HTTP enabled
# OR
encoderfile serve --disable-http  # Keep gRPC enabled
```

### Output not appearing
If output isn't visible, check:
1. Ensure you're not redirecting output to a file unintentionally
2. Check file permissions if using `--out-dir`
3. Verify the model is correctly configured

### Model type detection
The CLI automatically detects model type from configuration. If inference behaves unexpectedly:
1. Verify your model configuration
2. Ensure the model type matches your use case
3. Check model compatibility

## Complete Workflow Example

Here's a complete workflow from model export to deployment:

### Step 1: Export Model to ONNX

```bash
# Export a HuggingFace model to ONNX format
optimum-cli export onnx \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --task text-classification \
  ./models/sentiment-model
```

### Step 2: Create Configuration File

Create `sentiment-config.yml`:

```yaml
encoderfile:
  name: sentiment-analyzer
  version: "1.0.0"
  path: ./models/sentiment-model
  model_type: sequence_classification
  output_dir: ./build
```

### Step 3: Build Encoderfile Binary

```bash
# Build self-contained binary
./target/release/cli build -f sentiment-config.yml
```

This creates: `./build/sentiment-analyzer.encoderfile`

### Step 4: Run Inference

**Option A: Start server and use HTTP/gRPC**
```bash
# Start server
./build/sentiment-analyzer.encoderfile serve

# In another terminal - use HTTP API
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["This is amazing!", "This is terrible"]}'
```

**Option B: Direct CLI inference**
```bash
# Single inference
./build/sentiment-analyzer.encoderfile infer "This is amazing!"

# Batch inference
./build/sentiment-analyzer.encoderfile infer \
  "This is amazing!" \
  "This is terrible" \
  "This is okay" \
  -o results.json
```

### Step 5: Deploy

```bash
# Copy binary to deployment location
cp ./build/sentiment-analyzer.encoderfile /usr/local/bin/sentiment-analyzer

# The binary is self-contained - no dependencies needed!
sentiment-analyzer serve --http-port 8080
```

## Command Reference Summary

| Command | Tool | Purpose |
|---------|------|---------|
| `./target/release/cli build -f config.yml` | cli | Build self-contained binary from ONNX model |
| `./target/release/cli version` | cli | Print version information |
| `<model>.encoderfile serve` | encoderfile | Start HTTP/gRPC inference server |
| `<model>.encoderfile infer` | encoderfile | Run single inference from command line |
| `<model>.encoderfile mcp` | encoderfile | Start MCP server |

## Additional Resources

- [Getting Started Guide](getting-started.md) - Step-by-step tutorial
- [API Reference](api-reference.md) - HTTP/gRPC/MCP API documentation
- [BUILDING.md](../BUILDING.md) - Complete build guide with advanced configuration
- [GitHub Repository](https://github.com/mozilla-ai/encoderfile) - Source code and issues
