use super::Tensor;
use mlua::prelude::*;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn lp_normalize(&self, p: f32, axis: isize) -> Result<Self, LuaError> {
        if self.0.is_empty() {
            return Err(LuaError::external("Cannot normalize an empty tensor"));
        }
        if p == 0.0 {
            return Err(LuaError::external("p cannot equal 0.0"));
        }

        let axis = self.axis1(axis)?;
        let arr = &self.0;

        // Compute Lp norm along axis
        let norms = arr.map_axis(axis, |subview| {
            subview
                .iter()
                .map(|&v| v.abs().powf(p))
                .sum::<f32>()
                .powf(1.0 / p)
        });

        // Avoid division by zero using in-place broadcast clamp
        let norms = norms.mapv(|x| if x < 1e-12 { 1e-12 } else { x });

        // Broadcast division using ndarray’s broadcasting API
        let normalized = arr / &norms.insert_axis(axis);

        Ok(Self(normalized))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lp_norm_empty() {
        use ndarray::ArrayD;
        let arr: ArrayD<f32> = ndarray::array![[[]]].into_dyn();

        assert!(arr.is_empty());
        assert!(Tensor(arr).lp_normalize(1.0, 1).is_err())
    }

    #[test]
    fn test_lp_norm_zero() {
        use ndarray::{Array3, ArrayD};
        let arr: ArrayD<f32> = Array3::ones((3, 3, 3)).into_dyn();

        assert!(Tensor(arr).lp_normalize(0.0, 1).is_err())
    }

    #[test]
    fn test_lp_norm_nonexistent_dim() {
        use ndarray::{Array3, ArrayD};
        let arr: ArrayD<f32> = Array3::ones((3, 3, 3)).into_dyn();

        assert!(Tensor(arr.clone()).lp_normalize(1.0, 0).is_err()); // lua starts with 1
        assert!(Tensor(arr.clone()).lp_normalize(1.0, 4).is_err());
    }
}
