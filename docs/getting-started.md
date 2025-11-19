# Getting Started

This guide will help you set up your development environment and build your first encoderfile.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Rust](https://rust-lang.org/tools/install/) - For building the binary
- [Python 3.13+](https://www.python.org/downloads/) - For the build tools
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package manager
- [protoc](https://protobuf.dev/installation/) - Protocol Buffer compiler

### Installing Prerequisites

=== "macOS"

    ```bash
    # Install Rust
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install protoc
    brew install protobuf
    ```

=== "Linux"

    ```bash
    # Install Rust
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install protoc
    # Ubuntu/Debian
    sudo apt-get install protobuf-compiler

    # Fedora
    sudo dnf install protobuf-compiler
    ```

=== "Windows"

    ```powershell
    # Install Rust
    # Download and run rustup-init.exe from https://rustup.rs

    # Install uv
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

    # Install protoc
    # Download from https://github.com/protocolbuffers/protobuf/releases
    ```

## Development Setup

If you want to contribute to encoderfile or modify the source code:

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

!!! tip "VSCode Users"
    If you're using VSCode with the `rust-analyzer` plugin and encounter compilation errors, generate a default `.env` file:

    ```bash
    uv run -m encoderbuild.utils.create_dummy_env_file > .env
    ```

## Your First Encoderfile

Let's build a sentiment analysis model as an example.

### Step 1: Export a Model to ONNX

First, we need to export a HuggingFace model to ONNX format:

```bash
optimum-cli export onnx \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --task text-classification \
  ./sentiment-model
```

This will create a directory with the following structure:

```
sentiment-model/
├── config.json
├── model.onnx                # ONNX weights
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json            # Tokenizer
└── vocab.txt
```

!!! note "Task Types"
    Available tasks:

    - `feature-extraction` - For embedding models
    - `text-classification` - For sequence classification
    - `token-classification` - For NER/token tagging

    See the [HuggingFace task guide](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model) for more options.

### Step 2: Create Configuration File

Create a YAML configuration file `sentiment-config.yml`:

```yaml
encoderfile:
  name: sentiment-analyzer
  version: "1.0.0"
  path: ./sentiment-model
  model_type: sequence_classification
  output_dir: ./build
```

**Configuration fields:**

- `name` - Model identifier (used in API responses)
- `version` - Model version (optional, defaults to "0.1.0")
- `path` - Path to the model directory with ONNX weights
- `model_type` - Model type: `embedding`, `sequence_classification`, or `token_classification`
- `output_dir` - Where to output the binary (optional, defaults to current directory)

### Step 3: Build the Binary

First, build the CLI tool if you haven't already:

```bash
cargo build -p encoderfile --bin cli --release
```

Now build your encoderfile:

```bash
./target/release/cli build -f sentiment-config.yml
```

The binary will be created at `./build/sentiment-analyzer.encoderfile`.

### Step 4: Run the Server

Start the encoderfile server:

```bash
# Make the binary executable (Unix-like systems)
chmod +x ./build/sentiment-analyzer.encoderfile

# Start the server
./build/sentiment-analyzer.encoderfile serve
```

You should see output indicating the server is running:

```
Starting HTTP server on 0.0.0.0:8080
Starting gRPC server on [::]:50051
```

### Step 5: Make Predictions

Test your encoderfile with curl:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      "This product is amazing!",
      "Terrible experience, very disappointed"
    ]
  }'
```

Expected response:

```json
{
  "results": [
    {
      "logits": [-4.123, 4.567],
      "scores": [0.0001, 0.9999],
      "predicted_index": 1,
      "predicted_label": "POSITIVE"
    },
    {
      "logits": [4.234, -3.987],
      "scores": [0.9998, 0.0002],
      "predicted_index": 0,
      "predicted_label": "NEGATIVE"
    }
  ],
  "model_id": "sentiment-analyzer"
}
```

## Common Model Types

### Embedding Models

For text similarity, semantic search, and retrieval:

```bash
# Export
optimum-cli export onnx \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --task feature-extraction \
  ./embedding-model

# Create config
cat > embedding-config.yml <<EOF
encoderfile:
  name: embedder
  path: ./embedding-model
  model_type: embedding
  output_dir: ./build
EOF

# Build
./target/release/cli build -f embedding-config.yml

# Use
./build/embedder.encoderfile serve &
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello world"], "normalize": true}'
```

### Token Classification Models

For Named Entity Recognition (NER):

```bash
# Export
optimum-cli export onnx \
  --model dslim/bert-base-NER \
  --task token-classification \
  ./ner-model

# Create config
cat > ner-config.yml <<EOF
encoderfile:
  name: ner
  path: ./ner-model
  model_type: token_classification
  output_dir: ./build
EOF

# Build
./target/release/cli build -f ner-config.yml

# Use
./build/ner.encoderfile serve &
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Apple Inc. is in Cupertino, California"]}'
```

## Server Configuration

### Customizing Ports

```bash
# Custom HTTP port
./build/<model>.encoderfile serve --http-port 3000

# Custom gRPC port
./build/<model>.encoderfile serve --grpc-port 50052

# Both custom
./build/<model>.encoderfile serve --http-port 3000 --grpc-port 50052
```

### Disabling Services

```bash
# HTTP only
./build/<model>.encoderfile serve --disable-grpc

# gRPC only
./build/<model>.encoderfile serve --disable-http
```

### Using Custom Hostnames

```bash
./build/<model>.encoderfile serve \
  --http-hostname 127.0.0.1 \
  --grpc-hostname localhost
```

## CLI Inference

For one-off predictions without running a server:

```bash
# Single input
./build/<model>.encoderfile infer "This is a test sentence"

# Multiple inputs
./build/<model>.encoderfile infer "First text" "Second text" "Third text"

# Save to file
./build/<model>.encoderfile infer "Test" -o results.json
```

## Using Pre-Exported Models

Some models on HuggingFace already have ONNX weights:

```bash
# Clone a model with ONNX weights
git clone https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english

# Create config
cat > sentiment-config.yml <<EOF
encoderfile:
  name: sentiment
  path: ./distilbert-base-uncased-finetuned-sst-2-english
  model_type: sequence_classification
  output_dir: ./build
EOF

# Build directly
./target/release/cli build -f sentiment-config.yml
```

## Troubleshooting

### ONNX Export Fails

If `optimum-cli export` fails:

1. Check model compatibility - must be encoder-only
2. Try a different task type
3. Check the model's HuggingFace page for known issues

### Build Fails

If the build fails:

1. Ensure all prerequisites are installed
2. Check that the model directory has `model.onnx` and `tokenizer.json`
3. Verify the model type matches the architecture

### Server Won't Start

If the server fails to start:

1. Check if ports are already in use
2. Try different ports with `--http-port` and `--grpc-port`
3. Check file permissions on the binary

### Inference Errors

If inference fails:

1. Check input format matches the expected schema
2. Verify the server is running with `/health` endpoint
3. Check server logs for error messages

## Next Steps

Now that you have encoderfile running, explore:

- [**Building Guide**](building.md) - Advanced build options and optimization
- [**CLI Reference**](cli.md) - Complete command-line documentation
- [**API Reference**](api-reference.md) - HTTP, gRPC, and MCP APIs
- [**Contributing**](CONTRIBUTING.md) - Help improve encoderfile
