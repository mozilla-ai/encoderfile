# Transforms

Transforms allow you to post-process model outputs after ONNX inference and before returning results. They run inside the model binary, operating directly on tensors for high performance.

Transforms run on Lua 5.4 in a sandboxed environment. The transforms feature does not support LuaJIT currently.

## Why Use Transforms?

Common use cases:

- **Normalize embeddings** for cosine similarity
- **Apply softmax** to convert logits to probabilities
- **Pool embeddings** to create sentence representations
- **Scale outputs** for specific downstream tasks

## Getting Started

A transform is a Lua script that defines a `Postprocess` function:

```lua
---@param arr Tensor
---@return Tensor
function Postprocess(arr, ...)
    -- your postprocessing logic
    return tensor
end
```

With a handful of exceptions, the `Postprocess` function must return a `Tensor` with the exact same shape as the input `Tensor` provided for that model type. The exceptions are as follows:

- Embedding and sentence embedding models can modify the length of `hidden` (useful for matryoshka embeddings)
- Sentence embeddings are given a `Tensor` of shape `[batch_size, seq_len, hidden]` and attention mask of `[batch_size, seq_len]`, and must return a `Tensor` of shape `[batch_size, hidden]`. In other words, it expects a pooling operation along dimension `seq_len`.

!!! note "Note on indexing"
    Lua is 1-indexed, meaning that it starts counting at 1 instead of 0. The `Tensor` API reflects this, meaning that you must count your axes and indices starting at 1 instead of 0.

We provide a built-in API for standard tensor operations. To learn more, check out our [Tensor API reference page](reference.md). You can find the stub file [here](https://github.com/mozilla-ai/encoderfile/blob/main/encoderfile/stubs/lua/tensor.lua).

If you don't see an op that you need, please don't hesitate to [create an issue](https://github.com/mozilla-ai/encoderfile/issues) on Github.

## Creating a New Transform

To create a new transform, use the encoderfile CLI:

```
encoderfile new-transform --model-type [embedding|sequence_classification|etc.] > /path/to/your/transform/file.lua
```

## Input Signatures

The input signature of `Postprocess` depends on the type of model being used.

### Embedding

```lua
--- input: 3d tensor of shape [batch_size, seq_len, hidden]
---@param arr Tensor
---output: 3d tensor of shape [batch_size, seq_len, hidden]
---@return Tensor
function Postprocess(arr)
    -- your postprocessing logic
    return tensor
end
```

### Sequence Classification

```lua
--- input: 2d tensor of shape [batch_size, n_labels]
---@param arr Tensor
---output: 2d tensor of shape [batch_size, n_labels]
---@return Tensor
function Postprocess(arr)
    -- your postprocessing logic
    return tensor
end
```

### Token Classification

```lua
--- input: 3d tensor of shape [batch_size, seq_len, n_labels]
---@param arr Tensor
---output: 3d tensor of shape [batch_size, seq_len, n_labels]
---@return Tensor
function Postprocess(arr)
    -- your postprocessing logic
    return tensor
end
```

### Sentence Embedding


!!! note "Mean Pooling"
    To mean-pool embeddings, you can use the `Tensor:mean_pool` function like this: `tensor:mean_pool(mask)`.

```lua
--- input: 3d tensor of shape [batch_size, seq_len, hidden]
---@param arr Tensor
-- input: 2d tensor of shape [batch_size, seq_len]
-- This is automatically provided to the function and is equivalent to 🤗 transformer's attention_mask.
---@param mask Tensor
---output: 2d tensor of shape [batch_size, hidden]
---@return Tensor
function Postprocess(arr, mask)
    -- your postprocessing logic
    return tensor
end
```

## Typical Transform Patterns

Most transforms fall into one of 3 patterns:

### 1. Elementwise Transforms

Safe: they preserve shape automatically.

Examples:

- scaling (`tensor * 1.5`)
- activation functions (`tensor:exp()`)

### 2. Normalization Across Axis

These also preserve shape.

Examples:

- Lp normalization: (`tensor:lp_normalize(p, axis)`)
- subtracting mean per batch or per token
- applying softmax across a specific dimension (`tensor:softmax(2)`)

### 3. Mask-aware adjustments

When working with sentence embedding models:

```lua
function Postprocess(arr, mask)
    -- embeddings: [batch, seq, hidden]
    -- mask: [batch, seq]

    -- operations here must output [batch, hidden]
    return ...
end
```

## Best Practices

!!! warning "Performance Implications"
    Transforms run synchronously during inference, so expensive Lua-side loops will increase latency. If you don't see an op that you need, please don't hesitate to [create an issue](https://github.com/mozilla-ai/encoderfile/issues) on Github.

A typical transform follows this structure:

```lua
function Postprocess(arr, ...)
    -- Step 1: apply elementwise or axis-based operations
    local modified = arr:exp()  -- example

    -- Step 2: ensure the output shape matches the input shape
    -- (all built-in ops described in the Tensor API preserve shape)

    return modified
end
```

## Debugging Transforms

You can inspect shape and values using:

```lua
print("ndim:", t:ndim())
print("len:", #t)
print(tostring(t))
```

Errors typically fall into:

- axis out of range
    → axis must be 1-indexed and ≤ tensor rank

- broadcasting errors
    → the two shapes are incompatible

- returned value is not a tensor
    → must return a Tensor userdata object

- shape mismatch
    → you modified rank or dimensions

## Configuration

Transforms are embedded at build time. You can specify them in your config.yml either as a file path or inline.

```yml
transform:
    path: path/to/your/transform/here
```

Or, they can be passed inline:
```yml
transform: |
    function Postprocess(arr)
        ...
    return arr
end
```

