use super::Tensor;
use mlua::prelude::*;
use ndarray::ArrayD;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn clamp(&self, min: Option<f32>, max: Option<f32>) -> Result<Self, LuaError> {
        let input = self
            .0
            .as_slice()
            .ok_or_else(|| LuaError::external("Array must be contiguous"))?;

        let mut out = ArrayD::<f32>::zeros(self.0.raw_dim());
        let out_slice = out
            .as_slice_mut()
            .ok_or_else(|| LuaError::external("Failed to fetch output slice"))?;

        // NaN bound policy: if any bound is NaN, everything becomes NaN. For IEEE-754 compliance :d
        if min.is_some_and(f32::is_nan) || max.is_some_and(f32::is_nan) {
            for dst in out_slice.iter_mut() {
                *dst = f32::NAN;
            }
            return Ok(Self(out));
        }

        match (min, max) {
            (Some(lo), Some(hi)) => {
                for (dst, &src) in out_slice.iter_mut().zip(input.iter()) {
                    *dst = src.max(lo).min(hi);
                }
            }
            (Some(lo), None) => {
                for (dst, &src) in out_slice.iter_mut().zip(input.iter()) {
                    *dst = src.max(lo);
                }
            }
            (None, Some(hi)) => {
                for (dst, &src) in out_slice.iter_mut().zip(input.iter()) {
                    *dst = src.min(hi);
                }
            }
            (None, None) => {
                out_slice.copy_from_slice(input);
            }
        }

        Ok(Self(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_correctness() {
        let tensor = Tensor(ndarray::array!([-5.0, -1.0, 0.0, 1.0, 5.0]).into_dyn());
        let result = tensor
            .clamp(Some(-1.0), Some(1.0))
            .expect("Failed to clamp");
        let expected = Tensor(ndarray::array!([-1.0, -1.0, 0.0, 1.0, 1.0]).into_dyn());
        assert_eq!(result.0, expected.0);
    }

    #[test]
    fn test_clamp_lower_bound_only() {
        let tensor = Tensor(ndarray::array!([-3.0, 0.0, 2.0]).into_dyn());
        let result = tensor
            .clamp(Some(0.0), None)
            .expect("Failed to clamp tensor");
        let expected = Tensor(ndarray::array!([0.0, 0.0, 2.0]).into_dyn());
        assert_eq!(result.0, expected.0);
    }

    #[test]
    fn test_clamp_upper_bound_only() {
        let tensor = Tensor(ndarray::array!([-3.0, 0.0, 2.0, 5.0]).into_dyn());
        let result = tensor
            .clamp(None, Some(2.0))
            .expect("Failed to clamp tensor");
        let expected = Tensor(ndarray::array!([-3.0, 0.0, 2.0, 2.0]).into_dyn());
        assert_eq!(result.0, expected.0);
    }

    #[test]
    fn test_clamp_infinite_bounds() {
        let tensor = Tensor(ndarray::array!([-3.0, 0.0, 2.0, 5.0]).into_dyn());
        let result = tensor
            .clamp(Some(f32::NEG_INFINITY), Some(f32::INFINITY))
            .expect("Failed to clamp tensor");
        let expected = Tensor(ndarray::array!([-3.0, 0.0, 2.0, 5.0]).into_dyn());
        assert_eq!(result.0, expected.0);
    }

    #[test]
    fn test_clamp_multidimensional() {
        let tensor =
            Tensor(ndarray::array!([[-3.0, 3.0], [0.0, 0.0], [2.0, 2.0], [5.0, 5.0]]).into_dyn());
        let expected_shape = tensor.0.shape().to_owned();

        let result = tensor
            .clamp(Some(0.0), Some(1.0))
            .expect("Failed to clamp tensor");

        let expected =
            Tensor(ndarray::array!([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]).into_dyn());

        assert_eq!(result.0.shape(), expected_shape.as_slice());
        assert_eq!(result.0, expected.0);
    }

    #[test]
    fn test_clamp_identity() {
        let tensor = Tensor(ndarray::array!([-3.0, 0.0, 2.0, 5.0]).into_dyn());
        let result = tensor.clamp(None, None).expect("Failed to clamp tensor");
        assert_eq!(result.0, tensor.0);
    }

    #[test]
    fn test_clamp_min_equals_max() {
        let tensor = Tensor(ndarray::array!([0.0, 3.0, 10.0]).into_dyn());
        let result = tensor
            .clamp(Some(3.0), Some(3.0))
            .expect("Failed to clamp tensor");
        let expected = Tensor(ndarray::array!([3.0, 3.0, 3.0]).into_dyn());
        assert_eq!(result.0, expected.0);
    }

    #[test]
    fn test_clamp_inverted_bounds() {
        let tensor = Tensor(ndarray::array!([0.0, 3.0, 10.0]).into_dyn());
        let result = tensor
            .clamp(Some(5.0), Some(2.0))
            .expect("Failed to clamp tensor");
        let expected = Tensor(ndarray::array!([2.0, 2.0, 2.0]).into_dyn());
        assert_eq!(result.0, expected.0);
    }

    #[test]
    fn test_clamp_nan() {
        // clamping with NaN bounds nuke the entire tensor. Just so that we have no surprises later ;)
        let tensor = Tensor(ndarray::array!([0.0, 3.0, 10.0]).into_dyn());
        let result = tensor
            .clamp(Some(f32::NAN), Some(f32::NAN))
            .expect("Failed to clamp tensor");
        let expected = Tensor(ndarray::array!([f32::NAN, f32::NAN, f32::NAN]).into_dyn());
        for (a, b) in result.0.iter().zip(expected.0.iter()) {
            assert!(a.is_nan() && b.is_nan());
        }
    }
}
