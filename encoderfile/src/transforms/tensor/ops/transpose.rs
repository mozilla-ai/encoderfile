use super::Tensor;
use mlua::prelude::*;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn transpose(&self) -> Result<Self, LuaError> {
        Ok(Self(self.0.t().to_owned()))
    }
}

#[test]
fn test_transpose() {
    use ndarray::ArrayD;
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0], [4.0, 5.0],].into_dyn();
    let transpose = arr.t().into_owned();

    assert_eq!(Tensor(arr).transpose().unwrap().0, transpose)
}
