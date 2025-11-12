use super::{Tensor, add, div, load_env, mul, sub};
use mlua::prelude::*;
use ndarray::{Array0, Array2, Array3, Axis};

#[test]
fn test_min() {
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
    let tensor = Tensor(Array2::ones((3, 3)).into_dyn());
    assert_eq!(tensor.max().unwrap(), 1.0);
}

#[test]
fn test_max_empty() {
    let tensor = Tensor(ndarray::array![[[]]].into_dyn());
    assert!(tensor.max().is_err())
}

#[test]
fn test_exp() {
    let arr = Array2::ones((3, 3)).into_dyn();
    let tensor = Tensor(arr.clone());
    assert_eq!(tensor.exp().unwrap(), Tensor(arr.mapv(f32::exp)));
}

#[test]
fn test_exp_empty() {
    let tensor = Tensor(ndarray::array![[[]]].into_dyn());
    let Tensor(exp) = tensor.exp().unwrap();
    assert!(exp.is_empty());
}

#[test]
fn test_len() {
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

macro_rules! generate_ops_test {
    ($mod_name:ident, $op:tt, $rust_fn:ident, $lua_op:expr) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn test_binding() {
                let lua = load_env();
                let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());
                let arr2 = arr1.clone();

                let gold_val = $rust_fn(
                    &arr1,
                    LuaValue::UserData(lua.create_userdata(arr2.clone()).unwrap())
                ).expect("Failed to compute");

                let result: Tensor = lua.load(format!("return function(x, y) return x {} y end", $lua_op))
                    .eval::<LuaFunction>()
                    .unwrap()
                    .call((arr1, arr2))
                    .expect("Binding failed");

                assert_eq!(result, gold_val);
            }

            #[test]
            fn test_tensor() {
                let lua = load_env();
                let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());
                let arr2 = arr1.clone();

                let val = LuaValue::UserData(lua.create_userdata(arr1.clone()).unwrap());
                let result = $rust_fn(&arr1, val).unwrap();

                let gold = &arr1.0 $op &arr2.0;

                assert_eq!(gold, result.0);
            }

            #[test]
            fn test_number() {
                let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());

                let gold_sum = &arr1.0 $op Array2::<f32>::from_elem((3, 3), 5.0);

                let result = $rust_fn(&arr1, LuaValue::Number(5.0)).unwrap();

                assert_eq!(gold_sum, result.0);
            }

            #[test]
            fn test_integer() {
                let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());

                let gold_sum = &arr1.0 $op Array2::<f32>::from_elem((3, 3), 5.0);

                let result = $rust_fn(&arr1, LuaValue::Integer(5)).unwrap();

                assert_eq!(gold_sum, result.0);
            }

            #[test]
            fn test_bad_dtype() {
                let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());

                let result: Result<Tensor, LuaError> = $rust_fn(&arr1, LuaValue::Boolean(false));

                assert!(result.is_err());
            }
        }
    }
}

generate_ops_test!(
    test_addition, +, add, "+"
);

generate_ops_test!(
    test_subtraction, -, sub, "-"
);

generate_ops_test!(
    test_multiplication, *, mul, "*"
);

generate_ops_test!(
    test_division, /, div, "/"
);

#[test]
fn test_eq_simple() {
    let lua = load_env();

    let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());
    let arr2 = arr1.clone();

    assert!(arr1 == arr2);

    let result: bool = lua
        .load("return function(x, y) return x == y end")
        .eval::<LuaFunction>()
        .unwrap()
        .call((arr1, arr2))
        .expect("Failed to evaluate");

    assert!(result);
}

#[test]
fn test_neq_simple() {
    let lua = load_env();

    let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());
    let arr2 = Tensor(Array2::<f32>::zeros((3, 3)).into_dyn());

    assert!(arr1 != arr2);

    let result: bool = lua
        .load("return function(x, y) return x == y end")
        .eval::<LuaFunction>()
        .unwrap()
        .call((arr1, arr2))
        .expect("Failed to evaluate");

    assert!(!result);
}

#[test]
fn test_to_string() {
    let lua = load_env();

    let vec = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());
    let vec_str_gold = vec.0.to_string();

    let vec_str: String = lua
        .globals()
        .get::<LuaFunction>("tostring")
        .unwrap()
        .call(vec)
        .unwrap();

    assert_eq!(vec_str, vec_str_gold);
}

#[test]
fn test_mean() {
    let tensor = Tensor(Array2::ones((3, 3)).into_dyn());

    assert_eq!(
        tensor.mean().expect("Failed to calculate mean"),
        tensor.0.mean()
    );
}

#[test]
fn test_std() {
    let tensor = Tensor(Array2::ones((3, 3)).into_dyn());

    assert_eq!(
        tensor.std(1.0).expect("Failed to calculate mean"),
        tensor.0.std(1.0)
    );
}

#[test]
fn test_sum() {
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
fn test_sum_axis_columns() {
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

#[test]
fn test_sum_with_lua_binding() {
    let lua = load_env();
    let tensor = Tensor(Array2::<f32>::from_elem((3, 3), 2.0).into_dyn());

    let func = lua
        .load("return function(x) return x:sum() end")
        .eval::<LuaFunction>()
        .unwrap();

    let result: f32 = func.call(tensor.clone()).unwrap();
    assert_eq!(result, tensor.sum().unwrap());
}

#[test]
fn test_map_axis_zero_transform() {
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
    let lua = load_env();
    let tensor = Tensor(Array3::<f32>::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as f32).into_dyn());

    let func = lua
        .load("return function(x) return x * 2 end")
        .eval::<LuaFunction>()
        .unwrap();

    let result = tensor.map_axis(3, func).expect("Failed to map axis");

    assert_eq!(result.0, tensor.0 * 2.0);
}

#[test]
fn fold_axis_sum_rows() -> LuaResult<()> {
    let lua = load_env();
    let arr = ndarray::array![[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]].into_dyn();
    let t = Tensor(arr);

    let func: LuaFunction = lua.load(r#"
        return function(acc, x) return acc + x end
    "#).eval()?;

    let res = t.fold_axis(1, 0.0, func)?; // fold each row
    let v = res.0.into_dimensionality::<ndarray::Ix1>().unwrap();

    assert_eq!(v.as_slice().unwrap(), &[6.0, 15.0]);
    Ok(())
}

#[test]
fn fold_axis_product() -> LuaResult<()> {
    let lua = Lua::new();
    let arr = ndarray::array![[1.0, 2.0],
                        [3.0, 4.0]].into_dyn();
    let t = Tensor(arr);

    let func: LuaFunction = lua.load(r#"
        return function(acc, x) return acc * x end
    "#).eval()?;

    let res = t.fold_axis(1, 1.0, func)?; // multiply across each row
    let v = res.0.into_dimensionality::<ndarray::Ix1>().unwrap();

    assert_eq!(v.as_slice().unwrap(), &[2.0, 12.0]);
    Ok(())
}
