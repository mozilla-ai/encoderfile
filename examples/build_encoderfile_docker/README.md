# Example: Building an Encoderfile for Sentence Embeddings

This example shows how to package [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) into a standalone Encoderfile binary using Docker.  
You end up with a minimal image that contains exactly one thing: a fast, dependency-free sentence-embedding server.

## 🏗️ Build the Docker Image

```bash
docker build -t bge-small-en-v1_5 .
````

This does three things:

1. Downloads the ONNX-exported model + tokenizer
2. Builds an `encoderfile` binary via `encoderfile build`
3. Produces a final minimal container containing only that binary

## 🚀 Run the Server

Expose both REST (8080) and gRPC (50051):

```bash
docker run -p 8080:8080 -p 50051:50051 bge-small-en-v1_5
```

By default the binary runs `serve`, so you’re immediately hosting sentence embeddings over both protocols.

## 🧠 `transform.lua`

Encoderfile supports small Lua hooks for post-processing using a small tensor library described in `encoderfile-core/stubs/lua/tensor.lua`.
For `BAAI/bge-small-en-v1.5`, we stick to the standard recipe: masked mean pooling followed by L2 normalization.

```lua
--- @param arr Tensor
--- @param mask Tensor
--- @return Tensor
function Postprocess(arr, mask)
    -- Mean-pool token embeddings using the attention mask
    local mean_pooled = arr:mean_pool(mask)

    -- L2-normalize across the embedding dimension (axis 2; Lua is 1-indexed)
    local l2_normalized = mean_pooled:lp_normalize(2, 2)

    return l2_normalized
end
```

## 📦 `encoderfile.yml`

Minimal config used for this example:

```yaml
encoderfile:
  name: BAAI/bge-small-en-v1.5
  model_type: sentence_embedding
  output_path: model.encoderfile
  path:
    model_weights_path: onnx/model.onnx
    tokenizer_path: tokenizer.json
    model_config_path: config.json
  transform:
    path: transform.lua
```

---

## 🐳 Dockerfile

```dockerfile
FROM ghcr.io/mozilla-ai/encoderfile:latest AS build

# install uv (required for huggingface_hub only)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /encoderfile

# huggingface CLI + model download
RUN uv venv && uv pip install huggingface_hub
COPY download_model.sh .
RUN sh download_model.sh

# build encoderfile
COPY encoderfile.yml transform.lua ./
RUN encoderfile build -f encoderfile.yml

# final minimal image
FROM gcr.io/distroless/cc as final
COPY --from=build /encoderfile/model.encoderfile /usr/local/bin/model.encoderfile
ENTRYPOINT ["/usr/local/bin/model.encoderfile"]
CMD ["serve"]
```

---

## 📥 `download_model.sh`

```bash
uvx hf download \
  BAAI/bge-small-en-v1.5 \
  onnx/model.onnx \
  tokenizer.json \
  config.json \
  --local-dir /encoderfile
```
