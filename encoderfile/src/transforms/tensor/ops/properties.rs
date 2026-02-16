use super::Tensor;
use mlua::prelude::*;
use ndarray_stats::QuantileExt;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn sum(&self) -> Result<f32, LuaError> {
        Ok(self.0.sum())
    }

    #[tracing::instrument(skip_all)]
    pub fn min(&self) -> Result<f32, LuaError> {
        self.0
            .min()
            .copied()
            .map_err(|e| LuaError::external(format!("Min max error: {e}")))
    }

    #[tracing::instrument(skip_all)]
    pub fn max(&self) -> Result<f32, LuaError> {
        self.0
            .max()
            .copied()
            .map_err(|e| LuaError::external(format!("Min max error: {e}")))
    }

    #[tracing::instrument(skip_all)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    // The lint does not understand that this is
    // a Lua method, so same rules will not apply.
    // But it doesn't hurt to have one anyway.
    // Maybe....
    // #[allow(clippy::len_without_is_empty)]
    #[tracing::instrument(skip_all)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[tracing::instrument(skip_all)]
    pub fn std(&self, ddof: f32) -> Result<f32, LuaError> {
        Ok(self.0.std(ddof))
    }

    #[tracing::instrument(skip_all)]
    pub fn mean(&self) -> Result<Option<f32>, LuaError> {
        Ok(self.0.mean())
    }

    #[tracing::instrument(skip_all)]
    pub fn ndim(&self) -> Result<usize, LuaError> {
        Ok(self.0.ndim())
    }
}

#[tracing::instrument(skip_all)]
pub fn is_broadcastable(a: &[usize], b: &[usize]) -> bool {
    let ndim = a.len().max(b.len());

    for i in 0..ndim {
        let ad = *a.get(a.len().wrapping_sub(i + 1)).unwrap_or(&1);
        let bd = *b.get(b.len().wrapping_sub(i + 1)).unwrap_or(&1);

        if ad != bd && ad != 1 && bd != 1 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min() {
        use ndarray::Array2;

        let tensor = Tensor(Array2::ones((3, 3)).into_dyn());
        assert_eq!(tensor.min().unwrap(), 1.0);
    }

    #[test]
    fn test_min_empty() {
        let tensor = Tensor(ndarray::array![[[]]].into_dyn());
        assert!(tensor.min().is_err())
    }

    #[test]
    fn test_max() {
        use ndarray::Array2;

        let tensor = Tensor(Array2::ones((3, 3)).into_dyn());
        assert_eq!(tensor.max().unwrap(), 1.0);
    }

    #[test]
    fn test_max_empty() {
        let tensor = Tensor(ndarray::array![[[]]].into_dyn());
        assert!(tensor.max().is_err())
    }

    #[test]
    fn test_len() {
        use crate::transforms::tensor::load_env;
        use ndarray::Array2;

        let lua = load_env();
        let tensor = Tensor(Array2::zeros((3, 3)).into_dyn());
        let tensor_len = tensor.len();

        let len = lua
            .load("return function(x) return #x end")
            .eval::<LuaFunction>()
            .expect("Bad function")
            .call::<usize>(tensor)
            .expect("Function failed");

        assert_eq!(tensor_len, len);
    }

    #[test]
    fn test_ndim() {
        use crate::transforms::tensor::load_env;
        use ndarray::Array2;

        let lua = load_env();
        let tensor = Tensor(Array2::zeros((3, 3)).into_dyn());

        let ndim = lua
            .load("return function(x) return x:ndim() end")
            .eval::<LuaFunction>()
            .unwrap()
            .call::<usize>(tensor)
            .unwrap();

        assert_eq!(ndim, 2);
    }

    #[test]
    fn test_ndim_0() {
        use crate::transforms::tensor::load_env;
        use ndarray::Array0;

        let lua = load_env();
        let tensor = Tensor(Array0::<f32>::zeros(()).into_dyn());

        let ndim = lua
            .load("return function(x) return x:ndim() end")
            .eval::<LuaFunction>()
            .unwrap()
            .call::<usize>(tensor)
            .unwrap();

        assert_eq!(ndim, 0);
    }

    #[test]
    fn test_mean() {
        use ndarray::Array2;

        let tensor = Tensor(Array2::ones((3, 3)).into_dyn());

        assert_eq!(
            tensor.mean().expect("Failed to calculate mean"),
            tensor.0.mean()
        );
    }

    #[test]
    fn test_std() {
        use ndarray::Array2;

        let tensor = Tensor(Array2::ones((3, 3)).into_dyn());

        assert_eq!(
            tensor.std(1.0).expect("Failed to calculate mean"),
            tensor.0.std(1.0)
        );
    }

    #[test]
    fn test_sum() {
        use ndarray::Array2;

        let tensor = Tensor(Array2::<f32>::from_elem((3, 3), 2.0).into_dyn());
        let expected = 2.0 * 9.0; // 3x3 of 2.0
        assert_eq!(tensor.sum().unwrap(), expected);
    }

    #[test]
    fn test_sum_empty() {
        let tensor = Tensor(ndarray::ArrayD::<f32>::zeros(vec![0]));
        assert_eq!(tensor.sum().unwrap(), 0.0);
    }

    #[test]
    fn test_sum_with_lua_binding() {
        use crate::transforms::tensor::load_env;
        use ndarray::Array2;

        let lua = load_env();
        let tensor = Tensor(Array2::<f32>::from_elem((3, 3), 2.0).into_dyn());

        let func = lua
            .load("return function(x) return x:sum() end")
            .eval::<LuaFunction>()
            .unwrap();

        let result: f32 = func.call(tensor.clone()).unwrap();
        assert_eq!(result, tensor.sum().unwrap());
    }
}
