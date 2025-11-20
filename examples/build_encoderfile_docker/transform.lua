function Postprocess(arr, mask)
    -- mean pool using mask
    local mean_pooled = arr:mean_pool(mask)

    -- l2 normalize across second axis
    -- (remember that lua is 1-indexed)
    local l2_normalized = mean_pooled:lp_normalize(2, 2)

    return l2_normalized
end
