# Encoderfile API Reference

## Overview

Encoderfile provides three API interfaces for model inference:
- **HTTP REST API** - JSON-based HTTP endpoints (default port: 8080)
- **gRPC API** - Protocol Buffer-based RPC service (default port: 50051)
- **MCP (Model Context Protocol)** - Integration with MCP-compatible systems

The available endpoints depend on the model type your encoderfile was built with:
- `embedding` - Extract token embeddings from text
- `sequence_classification` - Classify entire text sequences (e.g., sentiment analysis)
- `token_classification` - Classify individual tokens (e.g., Named Entity Recognition)

---

## HTTP REST API

All endpoints return JSON responses. Errors return appropriate HTTP status codes with error messages.

### Common Endpoints

These endpoints are available for all model types:

#### `GET /health`

Health check endpoint to verify the server is running.

**Response:**
```json
"OK!"
```

**Status Codes:**
- `200 OK` - Server is healthy

**Example:**
```bash
curl http://localhost:8080/health
```

---

#### `GET /model`

Returns metadata about the loaded model.

**Response:**
```json
{
  "model_id": "string",
  "model_type": "embedding" | "sequence_classification" | "token_classification",
  "id2label": {
    "0": "LABEL1",
    "1": "LABEL2"
  }
}
```

**Fields:**
- `model_id` (string) - The model identifier specified during build
- `model_type` (string) - Type of model loaded
- `id2label` (object, optional) - Label mappings for classification models (not present for embedding models)

**Status Codes:**
- `200 OK` - Successful

**Example:**
```bash
curl http://localhost:8080/model
```

**Example Response:**
```json
{
  "model_id": "sentiment-analyzer",
  "model_type": "sequence_classification",
  "id2label": {
    "0": "NEGATIVE",
    "1": "POSITIVE"
  }
}
```

---

#### `GET /openapi.json`

Returns the OpenAPI specification for the API.

**Response:**
- OpenAPI 3.0 JSON specification

**Status Codes:**
- `200 OK` - Successful

**Example:**
```bash
curl http://localhost:8080/openapi.json
```

---

### Embedding Models

#### `POST /predict`

Generate embeddings for input text sequences.

**Request Body:**
```json
{
  "inputs": ["string"],
  "normalize": boolean,
  "metadata": {
    "key": "value"
  }
}
```

**Fields:**
- `inputs` (array of strings, required) - Text sequences to embed
- `normalize` (boolean, required) - Whether to L2-normalize the embeddings
- `metadata` (object, optional) - Custom key-value pairs to include in response

**Response:**
```json
{
  "results": [
    {
      "embeddings": [
        {
          "embedding": [0.123, -0.456, 0.789, ...],
          "token_info": {
            "token": "string",
            "token_id": 101,
            "start": 0,
            "end": 5
          }
        }
      ]
    }
  ],
  "model_id": "string",
  "metadata": {
    "key": "value"
  }
}
```

**Response Fields:**
- `results` (array) - One result per input sequence
  - `embeddings` (array) - One embedding per token in the sequence
    - `embedding` (array of floats) - The embedding vector
    - `token_info` (object, optional) - Information about the token
      - `token` (string) - The token text
      - `token_id` (integer) - The token's vocabulary ID
      - `start` (integer) - Character offset where token starts
      - `end` (integer) - Character offset where token ends
- `model_id` (string) - The model identifier
- `metadata` (object, optional) - Custom metadata from request

**Status Codes:**
- `200 OK` - Successful
- `422 Unprocessable Entity` - Invalid input
- `500 Internal Server Error` - Server error

**Example:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["Hello world", "Encoderfile is fast"],
    "normalize": true
  }'
```

**Example Response:**
```json
{
  "results": [
    {
      "embeddings": [
        {
          "embedding": [0.023, -0.156, 0.089, ...],
          "token_info": {
            "token": "[CLS]",
            "token_id": 101,
            "start": 0,
            "end": 0
          }
        },
        {
          "embedding": [0.134, -0.267, 0.412, ...],
          "token_info": {
            "token": "hello",
            "token_id": 7592,
            "start": 0,
            "end": 5
          }
        },
        {
          "embedding": [0.098, -0.234, 0.567, ...],
          "token_info": {
            "token": "world",
            "token_id": 2088,
            "start": 6,
            "end": 11
          }
        }
      ]
    }
  ],
  "model_id": "my-embedder"
}
```

---

### Sequence Classification Models

#### `POST /predict`

Classify entire text sequences.

**Request Body:**
```json
{
  "inputs": ["string"],
  "metadata": {
    "key": "value"
  }
}
```

**Fields:**
- `inputs` (array of strings, required) - Text sequences to classify
- `metadata` (object, optional) - Custom key-value pairs to include in response

**Response:**
```json
{
  "results": [
    {
      "logits": [1.234, -0.567],
      "scores": [0.9876, 0.0124],
      "predicted_index": 0,
      "predicted_label": "POSITIVE"
    }
  ],
  "model_id": "string",
  "metadata": {
    "key": "value"
  }
}
```

**Response Fields:**
- `results` (array) - One result per input sequence
  - `logits` (array of floats) - Raw model outputs before softmax
  - `scores` (array of floats) - Probability scores after softmax (sum to 1.0)
  - `predicted_index` (integer) - Index of the highest-scoring class
  - `predicted_label` (string, optional) - Label corresponding to the predicted index (if model has label mappings)
- `model_id` (string) - The model identifier
- `metadata` (object, optional) - Custom metadata from request

**Status Codes:**
- `200 OK` - Successful
- `422 Unprocessable Entity` - Invalid input
- `500 Internal Server Error` - Server error

**Example:**
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

**Example Response:**
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

---

### Token Classification Models

#### `POST /predict`

Classify individual tokens in text sequences.

**Request Body:**
```json
{
  "inputs": ["string"],
  "metadata": {
    "key": "value"
  }
}
```

**Fields:**
- `inputs` (array of strings, required) - Text sequences to process
- `metadata` (object, optional) - Custom key-value pairs to include in response

**Response:**
```json
{
  "results": [
    {
      "tokens": [
        {
          "token_info": {
            "token": "string",
            "token_id": 101,
            "start": 0,
            "end": 5
          },
          "logits": [1.234, -0.567, 0.891],
          "scores": [0.45, 0.10, 0.45],
          "label": "B-PER",
          "score": 0.45
        }
      ]
    }
  ],
  "model_id": "string",
  "metadata": {
    "key": "value"
  }
}
```

**Response Fields:**
- `results` (array) - One result per input sequence
  - `tokens` (array) - One classification per token
    - `token_info` (object) - Information about the token
      - `token` (string) - The token text
      - `token_id` (integer) - The token's vocabulary ID
      - `start` (integer) - Character offset where token starts
      - `end` (integer) - Character offset where token ends
    - `logits` (array of floats) - Raw model outputs before softmax
    - `scores` (array of floats) - Probability scores after softmax (sum to 1.0)
    - `label` (string) - The predicted label for this token
    - `score` (float) - The probability score for the predicted label
- `model_id` (string) - The model identifier
- `metadata` (object, optional) - Custom metadata from request

**Status Codes:**
- `200 OK` - Successful
- `422 Unprocessable Entity` - Invalid input
- `500 Internal Server Error` - Server error

**Example:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["Apple Inc. is located in Cupertino, California"]
  }'
```

**Example Response:**
```json
{
  "results": [
    {
      "tokens": [
        {
          "token_info": {
            "token": "Apple",
            "token_id": 2624,
            "start": 0,
            "end": 5
          },
          "logits": [2.34, -1.23, 0.45, -0.67],
          "scores": [0.89, 0.03, 0.06, 0.02],
          "label": "B-ORG",
          "score": 0.89
        },
        {
          "token_info": {
            "token": "Inc.",
            "token_id": 4297,
            "start": 6,
            "end": 10
          },
          "logits": [1.87, -0.98, 0.23, -0.45],
          "scores": [0.78, 0.05, 0.15, 0.02],
          "label": "I-ORG",
          "score": 0.78
        },
        {
          "token_info": {
            "token": "Cupertino",
            "token_id": 17887,
            "start": 26,
            "end": 35
          },
          "logits": [-0.45, 2.67, -1.23, 0.89],
          "scores": [0.04, 0.82, 0.02, 0.12],
          "label": "B-LOC",
          "score": 0.82
        },
        {
          "token_info": {
            "token": "California",
            "token_id": 2662,
            "start": 37,
            "end": 47
          },
          "logits": [-0.67, 2.45, -0.98, 0.78],
          "scores": [0.05, 0.76, 0.04, 0.15],
          "label": "B-LOC",
          "score": 0.76
        }
      ]
    }
  ],
  "model_id": "ner-model"
}
```

---

## gRPC API

The gRPC API provides the same functionality as the HTTP REST API using [Protocol Buffers](../encoderfile-core/proto). Three services are available depending on your model type.

### Connection Details

- **Default hostname:** `[::]` (all interfaces)
- **Default port:** `50051`
- **Protocol:** gRPC (HTTP/2)

### Service Definitions

All proto files are located in `encoderfile/proto/`.

#### Common Service Methods

All three services implement these methods:

##### `GetModelMetadata`

Returns metadata about the loaded model.

**Request:** Empty (`GetModelMetadataRequest`)

**Response:**
```protobuf
message GetModelMetadataResponse {
  string model_id = 1;
  ModelType model_type = 2;
  map<uint32, string> id2label = 3;
}

enum ModelType {
  MODEL_TYPE_UNSPECIFIED = 0;
  EMBEDDING = 1;
  SEQUENCE_CLASSIFICATION = 2;
  TOKEN_CLASSIFICATION = 3;
}
```

---

### Embedding Service

**Service:** `encoderfile.Embedding`

#### `Predict`

Generate embeddings for input text sequences.

**Request:**
```protobuf
message EmbeddingRequest {
  repeated string inputs = 1;
  bool normalize = 2;
  map<string, string> metadata = 3;
}
```

**Response:**
```protobuf
message EmbeddingResponse {
  repeated TokenEmbeddingSequence results = 1;
  string model_id = 2;
  map<string, string> metadata = 3;
}

message TokenEmbeddingSequence {
  repeated TokenEmbedding embeddings = 1;
}

message TokenEmbedding {
  repeated float embedding = 1;
  token.TokenInfo token_info = 2;
}

message TokenInfo {
  string token = 1;
  uint32 token_id = 2;
  uint32 start = 3;
  uint32 end = 4;
}
```

**Example (grpcurl):**
```bash
grpcurl -plaintext \
  -d '{
    "inputs": ["Hello world"],
    "normalize": true
  }' \
  localhost:50051 \
  encoderfile.Embedding/Predict
```

---

### Sequence Classification Service

**Service:** `encoderfile.SequenceClassification`

#### `Predict`

Classify entire text sequences.

**Request:**
```protobuf
message SequenceClassificationRequest {
  repeated string inputs = 1;
  map<string, string> metadata = 2;
}
```

**Response:**
```protobuf
message SequenceClassificationResponse {
  repeated SequenceClassificationResult results = 1;
  string model_id = 2;
  map<string, string> metadata = 3;
}

message SequenceClassificationResult {
  repeated float logits = 1;
  repeated float scores = 2;
  uint32 predicted_index = 3;
  optional string predicted_label = 4;
}
```

**Example (grpcurl):**
```bash
grpcurl -plaintext \
  -d '{
    "inputs": ["This product is amazing!"]
  }' \
  localhost:50051 \
  encoderfile.SequenceClassification/Predict
```

---

### Token Classification Service

**Service:** `encoderfile.TokenClassification`

#### `Predict`

Classify individual tokens in text sequences.

**Request:**
```protobuf
message TokenClassificationRequest {
  repeated string inputs = 1;
  map<string, string> metadata = 2;
}
```

**Response:**
```protobuf
message TokenClassificationResponse {
  repeated TokenClassificationResult results = 1;
  string model_id = 2;
  map<string, string> metadata = 3;
}

message TokenClassificationResult {
  repeated TokenClassification tokens = 1;
}

message TokenClassification {
  token.TokenInfo token_info = 1;
  repeated float logits = 2;
  repeated float scores = 3;
  string label = 4;
  float score = 5;
}
```

**Example (grpcurl):**
```bash
grpcurl -plaintext \
  -d '{
    "inputs": ["Apple Inc. is in Cupertino"]
  }' \
  localhost:50051 \
  encoderfile.TokenClassification/Predict
```

---

### gRPC Error Codes

gRPC errors use standard status codes:

| Status Code | HTTP Equivalent | Description |
|-------------|-----------------|-------------|
| `INVALID_ARGUMENT` | 422 | Invalid input provided |
| `INTERNAL` | 500 | Internal server error or configuration error |

---

## MCP (Model Context Protocol)

Encoderfile supports Model Context Protocol, allowing integration with MCP-compatible systems.

### Connection Details

- **Endpoint:** `/mcp`
- **Transport:** HTTP-based MCP protocol
- **Port:** Same as HTTP server (default: 8080)

### MCP Tools

Each model type exposes a single tool via MCP:

#### Embedding Models

**Tool:** `run_encoder`

**Description:** "Performs embeddings for input text sequences."

**Parameters:** Same as HTTP `EmbeddingRequest`

**Returns:** Same as HTTP `EmbeddingResponse`

---

#### Sequence Classification Models

**Tool:** `run_encoder`

**Description:** "Performs sequence classification of input text sequences."

**Parameters:** Same as HTTP `SequenceClassificationRequest`

**Returns:** Same as HTTP `SequenceClassificationResponse`

---

#### Token Classification Models

**Tool:** `run_encoder`

**Description:** "Performs token classification of input text sequences."

**Parameters:** Same as HTTP `TokenClassificationRequest`

**Returns:** Same as HTTP `TokenClassificationResponse`

---

### MCP Server Information

When connected, the MCP server provides:

- **Protocol Version:** `2025-06-18`
- **Capabilities:** Tools only
- **Server Info:** Build environment details

### MCP Usage Example

To use with an MCP client:

```bash
# Start encoderfile with MCP support
./encoderfile serve

# Connect via MCP client at http://localhost:8080/mcp
```

---

## Error Handling

### Error Types

Encoderfile uses three error types:

| Error Type | HTTP Status | gRPC Status | MCP Error Code | Description |
|------------|-------------|-------------|----------------|-------------|
| `InputError` | 422 Unprocessable Entity | `INVALID_ARGUMENT` | `INVALID_REQUEST` | Invalid input data |
| `InternalError` | 500 Internal Server Error | `INTERNAL` | `INTERNAL_ERROR` | Runtime error |
| `ConfigError` | 500 Internal Server Error | `INTERNAL` | `INTERNAL_ERROR` | Configuration error |

### Error Response Format

#### HTTP REST

Errors return a plain text error message with the appropriate status code:

```
HTTP/1.1 422 Unprocessable Entity
Content-Type: text/plain

Invalid input: empty text sequence
```

#### gRPC

Errors return a `Status` object:

```protobuf
status {
  code: INVALID_ARGUMENT
  message: "Invalid input: empty text sequence"
}
```

#### MCP

Errors return an MCP error object:

```json
{
  "code": "INVALID_REQUEST",
  "message": "Invalid input: empty text sequence",
  "data": null
}
```

---

## Client Examples

### Python (HTTP)

```python
import requests

# Embedding example
response = requests.post(
    "http://localhost:8080/predict",
    json={
        "inputs": ["Hello world"],
        "normalize": True
    }
)
result = response.json()
print(result["results"][0]["embeddings"])
```

### Python (gRPC)

```python
import grpc
from generated import encoderfile_pb2, encoderfile_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = encoderfile_pb2_grpc.EmbeddingStub(channel)

request = encoderfile_pb2.EmbeddingRequest(
    inputs=["Hello world"],
    normalize=True
)
response = stub.Predict(request)
print(response.results)
```

### JavaScript (HTTP)

```javascript
const response = await fetch('http://localhost:8080/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    inputs: ['Hello world'],
    normalize: true
  })
});

const result = await response.json();
console.log(result.results);
```

### Go (gRPC)

```go
package main

import (
    "context"
    "log"

    "google.golang.org/grpc"
    pb "path/to/generated/proto"
)

func main() {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()

    client := pb.NewEmbeddingClient(conn)

    req := &pb.EmbeddingRequest{
        Inputs:    []string{"Hello world"},
        Normalize: true,
    }

    resp, err := client.Predict(context.Background(), req)
    if err != nil {
        log.Fatal(err)
    }

    log.Println(resp.Results)
}
```

### cURL (HTTP)

```bash
# Get model metadata
curl http://localhost:8080/model

# Embedding prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello world"], "normalize": true}'

# Sequence classification
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["This is great!"]}'

# Token classification
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["John lives in Paris"]}'
```

---

## Rate Limiting & Performance

### Batching

All endpoints support batch processing by providing multiple inputs in a single request:

```json
{
  "inputs": ["text 1", "text 2", "text 3", ...]
}
```

Batch processing is more efficient than multiple single requests.

### Concurrency

Encoderfile uses async I/O and can handle multiple concurrent requests. The exact concurrency limit depends on:
- Available system resources (CPU, memory)
- Model size and complexity
- Input sequence length

### Best Practices

1. **Batch requests** when processing multiple texts
2. **Reuse connections** (HTTP keep-alive, gRPC channel pooling)
3. **Set appropriate timeouts** for long sequences
4. **Monitor memory usage** with large batches or long sequences
5. **Use gRPC** for high-throughput scenarios (lower overhead than HTTP/JSON)

---

## See Also

- [CLI Documentation](cli.md) - Command-line interface reference
- [Getting Started](getting-started.md) - Getting started guide
- [Contributing Guide](CONTRIBUTING.md) - Development setup
