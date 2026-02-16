use super::Tensor;
use mlua::prelude::*;
use ndarray::Axis;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn truncate_axis(&self, axis: isize, len: usize) -> Result<Self, LuaError> {
        let axis = self.axis1(axis)?;

        let actual_len = self.0.len_of(axis).min(len);

        let mut slice_spec = Vec::with_capacity(self.0.ndim());

        for i in 0..self.0.ndim() {
            if Axis(i) == axis {
                slice_spec.push(ndarray::SliceInfoElem::Slice {
                    start: 0,
                    end: Some(actual_len as isize),
                    step: 1,
                });
            } else {
                slice_spec.push(ndarray::SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                });
            }
        }

        Ok(Tensor(self.0.slice(&slice_spec[..]).to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_axis_correctness() {
        use ndarray::Array3;
        let tensor = Tensor(Array3::from_elem([3, 3, 3], 1.0).into_dyn());

        // truncate along 2rd axis (3rd in lua land) to 2
        let result = tensor
            .truncate_axis(3, 2)
            .expect("Failed to truncate tensor");
        let expected = Tensor(Array3::from_elem([3, 3, 2], 1.0).into_dyn());

        assert_eq!(result, expected);
    }

    #[test]
    fn test_truncate_axis_out_of_bounds() {
        use ndarray::Array3;
        let tensor = Tensor(Array3::from_elem([3, 3, 3], 1.0).into_dyn());

        // should return the same thing
        let result = tensor
            .truncate_axis(3, 500)
            .expect("Failed to truncate tensor");
        let expected = Tensor(Array3::from_elem([3, 3, 3], 1.0).into_dyn());

        assert_eq!(result, expected);
    }
}
