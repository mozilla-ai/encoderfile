use super::Tensor;
use mlua::prelude::*;

impl Tensor {
    #[tracing::instrument]
    pub fn map_axis(&self, axis: isize, func: LuaFunction) -> Result<Self, LuaError> {
        let axis = self.axis1(axis)?;

        // Pre-size by number of subviews, NOT tensor length.
        let out_len = self.0.shape()[axis.0];
        let mut out = Vec::with_capacity(out_len);

        for subview in self.0.axis_iter(axis) {
            // Only ONE allocation: convert subview into Tensor for Lua
            let tensor_arg = Tensor(subview.to_owned().into_dyn());
            let mapped: Tensor = func.call(tensor_arg).map_err(LuaError::external)?;
            out.push(mapped.0); // store raw ArrayD, not Tensor
        }

        // Stack views without re-wrapping as Tensor
        let views: Vec<_> = out.iter().map(|a| a.view()).collect();

        let stacked = ndarray::stack(axis, &views)
            .map_err(|e| LuaError::external(format!("stack error: {e}")))?;

        Ok(Tensor(stacked))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_axis_zero_transform() {
        use crate::transforms::tensor::load_env;
        use ndarray::Array3;
        let lua = load_env();
        let tensor = Tensor(Array3::<f32>::from_elem((3, 6, 9), 1.0).into_dyn());

        let func = lua
            .load("return function(x) return x end")
            .eval::<LuaFunction>()
            .unwrap();

        let result = tensor.map_axis(3, func).expect("Failed to map axis");

        assert_eq!(tensor, result);
    }

    #[test]
    fn test_map_axis_double_values() {
        use crate::transforms::tensor::load_env;
        use ndarray::Array3;
        let lua = load_env();
        let tensor = Tensor(
            Array3::<f32>::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as f32).into_dyn(),
        );

        let func = lua
            .load("return function(x) return x * 2 end")
            .eval::<LuaFunction>()
            .unwrap();

        let result = tensor.map_axis(3, func).expect("Failed to map axis");

        assert_eq!(result.0, tensor.0 * 2.0);
    }
}
