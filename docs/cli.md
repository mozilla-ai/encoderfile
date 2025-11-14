# Encoderfile CLI Documentation

## Overview

Encoderfile is a command-line tool for running machine learning inference tasks using transformer models. It supports multiple model types including embeddings, sequence classification, and token classification. The tool can operate in two modes: as a server (HTTP/gRPC) or as a standalone inference tool.

## Architecture

The CLI is built with the following components:
- **Server Mode**: Hosts models via HTTP and/or gRPC endpoints
- **Inference Mode**: Performs one-off inference operations from the command line
- **Multi-Model Support**: Automatically detects and routes to the appropriate model type

## Commands

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
| `--normalize` | Boolean | `true` | Normalize embeddings (embedding models only) |
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

## Additional Resources

- GitHub Repository: [mozilla-ai/encoderfile](https://github.com/mozilla-ai/encoderfile)
- API Documentation: Refer to the server API docs for HTTP/gRPC endpoint specifications
- Model Configuration: See configuration documentation for model setup
