-- Tensor type stubs (for IDE/LSP support)

---@diagnostic disable:missing-return

---@class Tensor
---@overload fun(tbl: table): Tensor
Tensor = {}

---Constructs a Tensor from a nested Lua table.
---The table must represent a rectangular n-dimensional array.
---@param tbl table Nested table of numbers
---@return Tensor
function Tensor.new(tbl) end

---Computes layer_norm along a specific axis
---@param axis integer Axis to compute layer_norm along
---@param eps number epsilon value
---@return Tensor
function Tensor:layer_norm(axis, eps) end

---Truncates a tensor along a specific axis.
---@param axis integer Axis to truncate along
---@param len integer Length to truncate each slice to
---@return Tensor
function Tensor:truncate_axis(axis, len) end

---Returns a new tensor with values clamped between `min` and `max`.
---If `min` is nil, no lower bound is applied.
---If `max` is nil, no upper bound is applied.
---Equivalent to `torch.clamp`.
---@param min number|nil Lower bound (optional)
---@param max number|nil Upper bound (optional)
---@return Tensor
function Tensor:clamp(min, max) end

---Computes the standard deviation of all elements.
---`ddof` specifies the degrees-of-freedom adjustment.
---@param ddof integer
---@return number
function Tensor:std(ddof) end

---Computes the arithmetic mean of all elements.
---@return number|nil Mean value, or nil if the tensor is empty
function Tensor:mean() end

---Returns the number of dimensions (rank) of the tensor.
---@return integer
function Tensor:ndim() end

---Computes the softmax along the specified axis.
---The result is normalized so values along that axis sum to 1.
---@param axis integer Axis index (1-based)
---@return Tensor
function Tensor:softmax(axis) end

---Returns a version of the tensor with the last two axes swapped.
---@return Tensor
function Tensor:transpose() end

---Normalizes values along an axis using the Lp norm.
---Each slice is divided by its Lp norm so that its magnitude becomes 1.
---@param lp number Norm order (e.g., 1 or 2)
---@param axis integer Axis index (1-based)
---@return Tensor
function Tensor:lp_normalize(lp, axis) end

---Returns the minimum scalar value in the tensor.
---@return number
function Tensor:min() end

---Returns the maximum scalar value in the tensor.
---@return number
function Tensor:max() end

---Applies the exponential function elementwise.
---@return Tensor
function Tensor:exp() end

---Sums values along the specified axis.
---@param axis integer Axis index (1-based)
---@return Tensor Tensor with the axis removed
function Tensor:sum_axis(axis) end

---Returns the sum of all elements in the tensor.
---@return number
function Tensor:sum() end

---Applies a function to each slice along an axis.
---`func` receives a Tensor containing one slice and must return a Tensor.
---@param axis integer Axis index (1-based)
---@param func fun(t: Tensor): Tensor
---@return Tensor
function Tensor:map_axis(axis, func) end

---Reduces each slice along an axis using a binary function.
---The function is called as `func(accumulator, value)` for each scalar.
---@param axis integer Axis index (1-based)
---@param func fun(acc: number, x: number): number
---@return Tensor 1-D tensor of reduction results
function Tensor:fold_axis(axis, func) end

---Mean pools a tensor using a mask.
---The mask must be 1 rank smaller than the tensor itself.
---@param mask Tensor Mask tensor
---@return Tensor
function Tensor:mean_pool(mask) end

---Elementwise equality comparison.
---@param other number|Tensor
---@return boolean
function Tensor:__eq(other) end

---Returns the total number of elements in the tensor.
---@return integer
function Tensor:__len() end

---Elementwise addition or broadcasting addition.
---@param other number|Tensor
---@return Tensor
function Tensor:__add(other) end

---Elementwise subtraction or broadcasting subtraction.
---@param other number|Tensor
---@return Tensor
function Tensor:__sub(other) end

---Elementwise multiplication or broadcasting multiplication.
---@param other number|Tensor
---@return Tensor
function Tensor:__mul(other) end

---Elementwise division or broadcasting division.
---@param other number|Tensor
---@return Tensor
function Tensor:__div(other) end

---Converts the tensor into a human-readable string representation.
---@return string
function Tensor:__tostring() end
