# Token Classification: Named Entity Recognition

This cookbook walks through building, deploying, and using a Named Entity Recognition (NER) model with Encoderfile. We'll use BERT fine-tuned for NER to identify people, organizations, and locations in text.

## What You'll Learn

- Export a token classification model to ONNX
- Build a self-contained encoderfile binary
- Deploy as a REST API server
- Make predictions via HTTP
- Use CLI for batch processing

## Prerequisites

- `encoderfile` CLI tool installed ([Installation Guide](../index.md#-installation))
- Python with `optimum[exporters]` for ONNX export
- `curl` for testing the API

---

## Step 1: Export the Model

We'll use `dslim/bert-base-NER`, a BERT model fine-tuned for named entity recognition.

!!! info "About the Model"
    This model recognizes 4 entity types:

    - **PER** - Person names
    - **ORG** - Organizations
    - **LOC** - Locations
    - **MISC** - Miscellaneous entities

### Export to ONNX

```bash
# Install optimum if you haven't already
pip install optimum[exporters]

# Export the model
optimum-cli export onnx \
  --model dslim/bert-base-NER \
  --task token-classification \
  ./ner-model
```

??? question "What files are created?"
    The export creates:
    ```
    ner-model/
    ├── config.json          # Model configuration
    ├── model.onnx          # ONNX weights
    ├── tokenizer.json      # Fast tokenizer
    ├── tokenizer_config.json
    └── special_tokens_map.json
    ```

---

## Step 2: Create Configuration

Create a YAML configuration file for building the encoderfile.

=== "ner-config.yml"

    ```yaml
    encoderfile:
      name: ner-tagger
      version: "1.0.0"
      path: ./ner-model
      model_type: token_classification
      output_path: ./build/ner-tagger.encoderfile
    ```

=== "With Optional Transform"

    ```yaml
    encoderfile:
      name: ner-tagger
      version: "1.0.0"
      path: ./ner-model
      model_type: token_classification
      output_path: ./build/ner-tagger.encoderfile
      transform: |
        --- Apply softmax to normalize logits
        function Postprocess(arr)
            return arr:softmax(3)
        end
    ```

!!! tip "Configuration Options"
    - `name` - Model identifier used in API responses
    - `path` - Directory containing ONNX model files
    - `model_type` - Must be `token_classification` for NER
    - `output_path` - Where to save the binary (optional)
    - `transform` - Optional Lua script for post-processing

---

## Step 3: Build the Binary

Build your self-contained encoderfile binary:

```bash
# Create output directory
mkdir -p build

# Build the encoderfile
encoderfile build -f ner-config.yml
```

!!! success "Build Output"
    You should see output like:
    ```
    Validating model...
    Generating project...
    Compiling binary...
    ✓ Build complete: ./build/ner-tagger.encoderfile
    ```

The resulting binary is **completely self-contained** - it includes:

- ONNX model weights
- Tokenizer
- Full inference runtime
- REST and gRPC servers

---

## Step 4: Start the Server

Launch the encoderfile server:

```bash
# Make executable (if needed)
chmod +x ./build/ner-tagger.encoderfile

# Start server
./build/ner-tagger.encoderfile serve
```

??? info "Server Startup"
    ```
    Starting HTTP server on 0.0.0.0:8080
    Starting gRPC server on [::]:50051
    Model: ner-tagger v1.0.0
    ```

The server is now running with both HTTP and gRPC endpoints.

---

## Step 5: Make Predictions

Now let's test the NER model with different types of text.

### Example 1: Basic Entity Recognition

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/predict \
      -H "Content-Type: application/json" \
      -d '{
        "inputs": ["Mozilla is headquartered in San Fancisco, CA"]
      }'
    ```

=== "Expected Response"

    ```json
    {
      "results": [
        {
          tokens: [{
           "token_info": {
                        "token": "Mozilla",
                        "token_id": 12556,
                        "start": 0,
                        "end": 2
                    },
                    "scores": [
                        -0.48987845,
                        2.912971,
                        -1.6960273,
                        2.2318482,
                        -3.2153757
                      ]
                  .....
          "label": "B-ORG",
          "score": 4.5583587
        }
        ]
        }
      ],
      "model_id": "ner-tagger"
    }
    ```

=== "Interpretation"

    **Entities Found:**

    - **Mozilla** → `B-ORG`, `I-ORG` (Organization)
    - **San Francisco** → `B-LOC` (Location)
    - **CA** → `B-LOC` (Location)

    The `B-` prefix indicates the beginning of an entity, `I-` indicates inside/continuation, and `O` means outside any entity.

### Example 2: Multiple Sentences

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/predict \
      -H "Content-Type: application/json" \
      -d '{
        "inputs": [
          "Yvon Chouinard founded Patagonia in 1957.",
          "The Eiffel Tower is located in Paris, France."
        ]
      }'
    ```

=== "Expected Entities"

    **Sentence 1:**
    - **Yvon** → Person (PER)
    - **Patagonia** → Organization (ORG)

    **Sentence 2:**
    - **Eiffel Tower** → Miscellaneous (MISC)
    - **Paris** → Location (LOC)
    - **France** → Location (LOC)

## Step 6: CLI Inference

For batch processing or one-off predictions, use the CLI directly:

### Single Input

```bash
./build/ner-tagger.encoderfile infer \
  "Tim Cook presented the new iPhone at Apple Park in California."
```

### Batch Processing

```bash
./build/ner-tagger.encoderfile infer \
  "Amazon was founded by Jeff Bezos in Seattle." \
  "Mozilla's headquarters are in San Francisco, California." \
  "Marie Curie won the Nobel Prize in Physics." \
  -o results.json
```

This saves all results to `results.json` for further processing.

---

## Advanced Usage

### Custom Ports

```bash
./build/ner-tagger.encoderfile serve \
  --http-port 3000 \
  --grpc-port 50052
```

### HTTP Only (Disable gRPC)

```bash
./build/ner-tagger.encoderfile serve --disable-grpc
```

### Production Deployment

```bash
# Copy to system location
sudo cp ./build/ner-tagger.encoderfile /usr/local/bin/

# Run as a service (example with systemd)
/usr/local/bin/ner-tagger.encoderfile serve \
  --http-hostname 0.0.0.0 \
  --http-port 8080
```

---

## Understanding the Output

### Token Classification Labels

The model uses the IOB (Inside-Outside-Beginning) tagging scheme:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `B-` | Beginning of entity | `B-PER` for "Barack" in "Barack Obama" |
| `I-` | Inside/continuation | `I-PER` for "Obama" in "Barack Obama" |
| `O` | Outside any entity | `O` for "is" or "the" |

### Entity Types

| Label | Description | Examples |
|-------|-------------|----------|
| `PER` | Person names | "John Smith", "Marie Curie" |
| `ORG` | Organizations | "Apple Inc.", "United Nations" |
| `LOC` | Locations | "Paris", "California", "Mount Everest" |
| `MISC` | Miscellaneous | "iPhone", "Nobel Prize" |

### Response Format

```json
{
  "results": [
    {
      "tokens": ["word1", "word2", ...],          // Tokenized input
      "logits": [[...], [...], ...],              // Raw model outputs
      "predicted_labels": ["B-PER", "O", ...]     // Predicted entity tags
    }
  ],
  "model_id": "ner-tagger"
}
```

---

## Troubleshooting

### Unexpected Entity Recognition

!!! warning "Model Limitations"
    The model may struggle with:

    - Rare or domain-specific entities
    - Ambiguous contexts (e.g., "Washington" as person vs. location)
    - Non-English text
    - Very long sequences (>512 tokens)

**Solution:** Fine-tune on domain-specific data or use a specialized model.

### Performance Optimization

If inference is slow:

```yaml
# Consider adding a transform to reduce output size
transform: |
  function Postprocess(arr)
    -- Only return top prediction per token
    return arr:argmax(3)
  end
```

### Server Connection Issues

```bash
# Check if server is running
curl http://localhost:8080/health

# Try different port
./build/ner-tagger.encoderfile serve --http-port 8081
```

---

## Next Steps

- **[Sequence Classification Cookbook](sequence-classification-sentiment.md)** - Build a sentiment analyzer
- **[Embedding Cookbook](embeddings-similarity.md)** - Create a semantic search engine
- **[Transforms Reference](../transforms/reference.md)** - Learn about custom post-processing
- **[API Reference](../reference/api-reference.md)** - Complete API documentation

---

## Summary

You've learned to:

- ✅ Export a token classification model to ONNX
- ✅ Build a self-contained encoderfile binary
- ✅ Deploy as a REST API server
- ✅ Make predictions via HTTP and CLI
- ✅ Understand NER output format

The encoderfile you built is production-ready and can be deployed anywhere without dependencies!
