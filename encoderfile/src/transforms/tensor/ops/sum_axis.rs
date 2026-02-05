use mlua::prelude::*;
use super::Tensor;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn sum_axis(&self, axis: isize) -> Result<Self, LuaError> {
        Ok(Self(self.0.sum_axis(self.axis1(axis)?)))
    }
}

#[test]
fn test_sum_axis_columns() {
    use ndarray::{Array2, Axis};
    let tensor = Tensor(
        Array2::<f32>::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.])
            .unwrap()
            .into_dyn(),
    );
    let result = tensor.sum_axis(2).unwrap();
    let expected = Tensor(ndarray::array![6., 15.].into_dyn());
    assert_eq!(result, expected);

    let expected = tensor.0.sum_axis(Axis(1));
    assert_eq!(result, Tensor(expected));
}

#[test]
fn test_sum_axis_rows() {
    use ndarray::{Array2, Axis};
    let tensor = Tensor(
        Array2::<f32>::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.])
            .unwrap()
            .into_dyn(),
    );
    let result = tensor.sum_axis(1).unwrap();
    let expected = Tensor(ndarray::array![5., 7., 9.].into_dyn());
    assert_eq!(result, expected);

    let expected = tensor.0.sum_axis(Axis(0));
    assert_eq!(result, Tensor(expected));
}

#[test]
fn test_sum_axis_invalid() {
    use ndarray::Array2;
    let tensor = Tensor(
        Array2::<f32>::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.])
            .unwrap()
            .into_dyn(),
    );
    let result = tensor.sum_axis(3); // invalid axis (too large)
    assert!(result.is_err());
}

#[test]
fn test_sum_axis_with_lua_binding() {
    use ndarray::Array2;
    use crate::transforms::tensor::load_env
;
    let lua = load_env();
    let tensor = Tensor(
        Array2::<f32>::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.])
            .unwrap()
            .into_dyn(),
    );

    let func = lua
        .load("return function(x) return x:sum_axis(2) end")
        .eval::<LuaFunction>()
        .unwrap();

    let result: Tensor = func.call(tensor.clone()).unwrap();
    let expected = Tensor(ndarray::array![6., 15.].into_dyn());
    assert_eq!(result, expected);
}
