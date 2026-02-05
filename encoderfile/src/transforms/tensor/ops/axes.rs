use mlua::prelude::*;
use ndarray::Axis;
use super::Tensor;

impl Tensor {
    pub fn axis1(&self, axis: isize) -> Result<Axis, LuaError> {
        if axis <= 0 {
            return Err(LuaError::external("Axis must be >= 1."));
        }

        let axis_index = (axis - 1) as usize;

        if axis_index >= self.0.ndim() {
            return Err(LuaError::external("Axis out of range."));
        }

        Ok(Axis(axis_index))
    }
}
