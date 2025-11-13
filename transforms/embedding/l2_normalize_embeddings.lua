--- Applies L2 normalization across the embedding dimension.
--- Each token embedding is scaled to unit length independently.
---
--- Args:
---   arr (Tensor): A tensor of shape [batch_size, n_tokens, hidden_dim].
---                 Normalization is applied along the third axis (hidden_dim).
---
--- Returns:
---   Tensor: The input tensor with L2-normalized embeddings.
---@param arr Tensor
---@return Tensor
function Postprocess(arr)
    return arr:lp_normalize(2, 3)
end
