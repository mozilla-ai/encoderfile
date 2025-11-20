# Getting Started

This quick-start guide will help you build and run your first encoderfile in under 10 minutes.

## Prerequisites

### Encoderfile CLI Tool

You need the `encoderfile` CLI tool installed:

- **Pre-built binary** (Linux/macOS): Download from [releases](https://github.com/mozilla-ai/encoderfile/releases) (TODO: add actual release link)
- **Build from source** (all platforms): See [BUILDING.md](../BUILDING.md)

### Python with Optimum

For exporting models to ONNX:

```bash
pip install optimum[exporters]
```

## Your First Encoderfile

Let's build a sentiment analysis model as an example.

### Step 1: Export Model to ONNX

Export a HuggingFace model to ONNX format:

```bash
optimum-cli export onnx \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --task text-classification \
  ./sentiment-model
```

This creates a directory with the required files:

```
sentiment-model/
├── config.json
├── model.onnx                # ONNX weights
├── tokenizer.json            # Tokenizer
└── ... (other files)
```

**Available task types:**
- `feature-extraction` - For embedding models
- `text-classification` - For sequence classification
- `token-classification` - For NER/token tagging

### Step 2: Create Configuration File

Create `sentiment-config.yml`:

```yaml
encoderfile:
  name: sentiment-analyzer
  version: "1.0.0"
  path: ./sentiment-model
  model_type: sequence_classification
  output_path: ./build/sentiment-analyzer.encoderfile
```

**Key fields:**
- `name` - Model identifier (used in API responses)
- `path` - Path to the model directory with ONNX weights
- `model_type` - `embedding`, `sequence_classification`, or `token_classification`
- `output_path` - Where to output the binary (optional, defaults to `./<name>.encoderfile`)

### Step 3: Build the Binary

Build your encoderfile:

```bash
encoderfile build -f sentiment-config.yml
```

> **Note:** If you built the CLI from source, use: `./target/release/encoderfile build -f sentiment-config.yml`

The binary will be created at `./build/sentiment-analyzer.encoderfile`.

### Step 4: Run the Server

Start your encoderfile server:

```bash
chmod +x ./build/sentiment-analyzer.encoderfile
./build/sentiment-analyzer.encoderfile serve
```

You should see:
```
Starting HTTP server on 0.0.0.0:8080
Starting gRPC server on [::]:50051
```

### Step 5: Make Predictions

Test with curl:

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

## Quick Examples

### Embedding Model

```bash
# Export
optimum-cli export onnx \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --task feature-extraction \
  ./embedding-model

# Config
cat > embedding-config.yml <<EOF
encoderfile:
  name: embedder
  path: ./embedding-model
  model_type: embedding
  output_path: ./build/embedder.encoderfile
EOF

# Build
encoderfile build -f embedding-config.yml

# Run
./build/embedder.encoderfile serve
```

### Token Classification (NER)

```bash
# Export
optimum-cli export onnx \
  --model dslim/bert-base-NER \
  --task token-classification \
  ./ner-model

# Config
cat > ner-config.yml <<EOF
encoderfile:
  name: ner
  path: ./ner-model
  model_type: token_classification
  output_path: ./build/ner.encoderfile
EOF

# Build
encoderfile build -f ner-config.yml

# Run
./build/ner.encoderfile serve
```

## Common Tasks

### Server Configuration

**Custom ports:**
```bash
./build/my-model.encoderfile serve --http-port 3000 --grpc-port 50052
```

**HTTP only (disable gRPC):**
```bash
./build/my-model.encoderfile serve --disable-grpc
```

**gRPC only (disable HTTP):**
```bash
./build/my-model.encoderfile serve --disable-http
```

### CLI Inference

Run inference without starting a server:

```bash
# Single input
./build/my-model.encoderfile infer "Test sentence"

# Multiple inputs
./build/my-model.encoderfile infer "First" "Second" "Third"

# Save to file
./build/my-model.encoderfile infer "Test" -o results.json
```

### Using Pre-Exported Models

Some HuggingFace models already have ONNX weights:

```bash
# Clone model with existing ONNX weights
git clone https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english

# Build directly
cat > config.yml <<EOF
encoderfile:
  name: sentiment
  path: ./distilbert-base-uncased-finetuned-sst-2-english
  model_type: sequence_classification
  output_path: ./build/sentiment.encoderfile
EOF

encoderfile build -f config.yml
```

## Troubleshooting

### ONNX Export Fails

- Check model compatibility (must be encoder-only)
- Try a different task type
- Check the model's HuggingFace page for known issues

### Build Fails

- Ensure the model directory has `model.onnx`, `tokenizer.json`, and `config.json`
- Verify the model type matches the architecture
- See [BUILDING.md](../BUILDING.md) for detailed troubleshooting

### Server Won't Start

- Check if ports are already in use
- Try different ports with `--http-port` and `--grpc-port`
- Check file permissions: `chmod +x ./build/my-model.encoderfile`

### Inference Errors

- Check input format matches the expected schema
- Verify the server is running
- Check server logs for error messages

## Next Steps

- **[BUILDING.md](../BUILDING.md)** - Complete build guide with advanced configuration options
- **[CLI Reference](cli.md)** - Full command-line documentation
- **[API Reference](api-reference.md)** - REST, gRPC, and MCP API documentation
- **[Contributing](../CONTRIBUTING.md)** - Help improve encoderfile
