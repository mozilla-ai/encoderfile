use super::Tensor;
use mlua::prelude::*;
use ndarray::Array1;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn fold_axis(&self, axis: isize, acc: f32, func: LuaFunction) -> Result<Tensor, LuaError> {
        let axis = self.axis1(axis)?;

        let mut out = Vec::new();

        for subview in self.0.axis_iter(axis) {
            let mut acc = acc;

            for &x in subview.iter() {
                acc = func.call((acc, x)).map_err(LuaError::external)?;
            }

            out.push(acc);
        }

        let result = Array1::from_shape_vec(out.len(), out)
            .expect("Failed to recast results")
            .into_dyn();

        Ok(Tensor(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_axis_sum_rows() -> LuaResult<()> {
        use crate::transforms::tensor::load_env;
        let lua = load_env();
        let arr = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let t = Tensor(arr);

        let func: LuaFunction = lua
            .load(
                r#"
            return function(acc, x) return acc + x end
        "#,
            )
            .eval()?;

        let res = t.fold_axis(1, 0.0, func)?; // fold each row
        let v = res.0.into_dimensionality::<ndarray::Ix1>().unwrap();

        assert_eq!(v.as_slice().unwrap(), &[6.0, 15.0]);
        Ok(())
    }

    #[test]
    fn fold_axis_product() -> LuaResult<()> {
        use crate::transforms::tensor::load_env;
        let lua = load_env();
        let arr = ndarray::array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let t = Tensor(arr);

        let func: LuaFunction = lua
            .load(
                r#"
            return function(acc, x) return acc * x end
        "#,
            )
            .eval()?;

        let res = t.fold_axis(1, 1.0, func)?; // multiply across each row
        let v = res.0.into_dimensionality::<ndarray::Ix1>().unwrap();

        assert_eq!(v.as_slice().unwrap(), &[2.0, 12.0]);
        Ok(())
    }
}
