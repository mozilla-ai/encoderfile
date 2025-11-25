# Transforms

Transforms allow you to modify model outputs immediately after ONNX inference and before the result is returned from the Encoderfile server or CLI. They run inside the model binary and operate directly on the tensors produced by the model.

A transform is a Lua file or snippet that defines a function `Postprocess`:

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
    Lua is 1-indexed, meaning that it starts counting at 1 instead of 0. The `Tensor` API reflects this, meaning that you have to start counting your axes at 1 instead of 0.

We provide a built-in API for standard tensor operations. To learn more, check out our [Tensor API reference page](reference). You can find the stub file [here](https://github.com/mozilla-ai/encoderfile/blob/main/encoderfile-core/stubs/lua/tensor.lua).

If you don't see an op that you need, please don't hesitate to [create an issue](https://github.com/mozilla-ai/encoderfile/issues) on Github.

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

```lua
--- input: 3d tensor of shape [batch_size, seq_len, hidden]
---@param arr Tensor
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

## Embedding Transforms
Transforms are embedded at build time. They can either be passed as a path in the config.yml:

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
```

