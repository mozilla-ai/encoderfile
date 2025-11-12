-- Applies a softmax across token classification logits.
-- Each token classification is normalized independently.
-- 
-- Args:
--   arr (Tensor): A tensor of shape [batch_size, n_tokens, n_labels].
--                 The softmax is applied along the third axis (n_labels).
--
-- Returns:
--   Tensor: The input tensor with softmax-normalized embeddings.
function Postprocess(arr)
    return arr:softmax(3)
end