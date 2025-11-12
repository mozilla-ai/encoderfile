-- Applies a softmax across the label dimension.
-- Each sample's label scores are normalized into a probability distribution.
--
-- Args:
--   arr (Tensor): A tensor of shape [batch_size, num_labels].
--                 The softmax is applied along the second axis (num_labels).
--
-- Returns:
--   Tensor: The input tensor with softmax-normalized label scores.
function Postprocess(arr)
    return arr:softmax(2)
end