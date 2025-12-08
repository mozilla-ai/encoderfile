use super::{Tensor, add, div, load_env, mul, sub};
use mlua::prelude::*;
use ndarray::{Array0, Array2, Array3, Axis, array};

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
    let tensor =
        Tensor(Array3::<f32>::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as f32).into_dyn());

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
    let lua = Lua::new();
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

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor(ndarray::ArrayD::from_shape_vec(shape, data).unwrap())
}

fn lua_number(n: f64) -> LuaValue {
    LuaValue::Number(n)
}

fn lua_tensor(t: Tensor, lua: &Lua) -> LuaValue {
    mlua::Value::UserData(lua.create_userdata(t).unwrap())
}

#[test]
fn test_add_broadcast_success() {
    let lua = Lua::new();

    // (2, 3) + (3,) → OK via broadcasting
    let a = tensor(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
    let b = tensor(vec![10., 20., 30.], &[3]);

    let res = add(&a, lua_tensor(b, &lua)).unwrap();
    assert_eq!(
        res.0,
        ndarray::arr2(&[[11., 22., 33.], [14., 25., 36.]]).into_dyn()
    );
}

#[test]
fn test_add_broadcast_failure() {
    let lua = Lua::new();

    // (2, 3) + (2,) → NOT broadcastable because trailing dims mismatch
    let a = tensor(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
    let b = tensor(vec![1., 2.], &[2]);

    let err = add(&a, lua_tensor(b, &lua)).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("not broadcastable"), "Got: {msg}");
}

#[test]
fn test_sub_broadcast_success() {
    let lua = Lua::new();

    // (3, 1) - (3,) → OK (result is (3,3))
    let a = tensor(vec![1., 2., 3.], &[3, 1]);
    let b = tensor(vec![1., 10., 100.], &[3]);

    let res = sub(&a, lua_tensor(b, &lua)).unwrap();
    assert_eq!(
        res.0,
        ndarray::arr2(&[[0., -9., -99.], [1., -8., -98.], [2., -7., -97.]]).into_dyn()
    );
}

#[test]
fn test_sub_broadcast_failure() {
    let lua = Lua::new();

    // (3,2) - (3,) → failure: trailing dim (2 vs 3)
    let a = tensor(vec![1., 2., 3., 4., 5., 6.], &[3, 2]);
    let b = tensor(vec![1., 2., 3.], &[3]);

    let err = sub(&a, lua_tensor(b, &lua)).unwrap_err();
    assert!(format!("{err}").contains("not broadcastable"));
}

#[test]
fn test_mul_broadcast_success() {
    // (2,3) * scalar → always OK
    let a = tensor(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
    let res = mul(&a, lua_number(2.0)).unwrap();

    assert_eq!(
        res.0,
        ndarray::arr2(&[[2., 4., 6.], [8., 10., 12.]]).into_dyn()
    );
}

#[test]
fn test_mul_broadcast_shape_success() {
    let lua = Lua::new();

    // (4,1) * (1,3) → → (4,3)
    let a = tensor(vec![1., 2., 3., 4.], &[4, 1]);
    let b = tensor(vec![10., 20., 30.], &[1, 3]);

    let res = mul(&a, lua_tensor(b, &lua)).unwrap();

    assert_eq!(
        res.0,
        ndarray::arr2(&[
            [10., 20., 30.],
            [20., 40., 60.],
            [30., 60., 90.],
            [40., 80., 120.]
        ])
        .into_dyn()
    );
}

#[test]
fn test_mul_broadcast_fail() {
    let lua = Lua::new();

    // (2,2) * (3,) → cannot broadcast trailing dims
    let a = tensor(vec![1., 2., 3., 4.], &[2, 2]);
    let b = tensor(vec![1., 2., 3.], &[3]);

    let err = mul(&a, lua_tensor(b, &lua)).unwrap_err();
    assert!(format!("{err}").contains("not broadcastable"));
}

#[test]
fn test_div_broadcast_success() {
    let lua = Lua::new();

    // (3,3) / (3,) → OK
    let a = tensor((1..=9).map(|x| x as f32).collect(), &[3, 3]);
    let b = tensor(vec![1., 2., 3.], &[3]);

    let res = div(&a, lua_tensor(b, &lua)).unwrap();

    assert_eq!(
        res.0,
        ndarray::arr2(&[
            [1.0 / 1., 2.0 / 2., 3.0 / 3.],
            [4.0 / 1., 5.0 / 2., 6.0 / 3.],
            [7.0 / 1., 8.0 / 2., 9.0 / 3.],
        ])
        .into_dyn()
    );
}

#[test]
fn test_div_broadcast_fail() {
    let lua = Lua::new();

    // (2,3) vs (2,) again → nope
    let a = tensor(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
    let b = tensor(vec![1., 2.], &[2]);

    let err = div(&a, lua_tensor(b, &lua)).unwrap_err();
    assert!(format!("{err}").contains("not broadcastable"));
}

#[test]
fn mean_pool_single_vector_no_mask() {
    // shape: (batch=1, seq=1, dim=3)
    let x = Tensor(array![[[1.0, 2.0, 3.0]]].into_dyn());
    let mask = Tensor(array![[1.0]].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();
    assert_eq!(pooled.0, array![[1.0, 2.0, 3.0]].into_dyn());
}

#[test]
fn mean_pool_two_tokens_equal_weight() {
    // shape: (1, 2, 3)
    let x = Tensor(array![[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]].into_dyn());

    let mask = Tensor(array![[1.0, 1.0]].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();
    let expected = array![[2.0, 2.0, 2.0]].into_dyn();

    assert_allclose(&pooled.0, &expected);
}

#[test]
fn mean_pool_ignores_masked_tokens() {
    // shape: (1, 3, 2)
    // Only the first and last token should count.
    let x = Tensor(
        array![[
            [10.0, 0.0],
            [99.0, 99.0], // masked out
            [20.0, 0.0]
        ]]
        .into_dyn(),
    );

    let mask = Tensor(array![[1.0, 0.0, 1.0]].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();
    let expected = array![[(10.0 + 20.0) / 2.0, 0.0]].into_dyn();

    assert_allclose(&pooled.0, &expected);
}

#[test]
fn mean_pool_batch_mode() {
    // shape: (2, 2, 2)
    let x = Tensor(
        array![
            [[1.0, 1.0], [3.0, 3.0]], // batch 0
            [[2.0, 4.0], [4.0, 2.0]], // batch 1
        ]
        .into_dyn(),
    );

    let mask = Tensor(array![[1.0, 1.0], [1.0, 0.0],].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();

    let expected = array![[(1.0 + 3.0) / 2.0, (1.0 + 3.0) / 2.0], [2.0, 4.0]].into_dyn();

    assert_allclose(&pooled.0, &expected);
}

#[test]
fn mean_pool_mask_broadcasting() {
    let x = Tensor(
        array![[
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]
        ]]
        .into_dyn(),
    );

    let mask = Tensor(array![[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();

    // Compute manually:
    // First inner seq: avg of [1,2] and [4,5]
    // Second inner seq isn't separate — everything is reduced together.
    //
    // Values included:
    //   1.0, 2.0, 4.0, 5.0   (mask=1)
    // and the same duplicated for the second feature.
    let expected = array![[3.0, 3.0]].into_dyn(); // (1,2)

    assert_allclose(&pooled.0, &expected);
}

fn assert_allclose(a: &ndarray::ArrayD<f32>, b: &ndarray::ArrayD<f32>) {
    let tol = 1e-6;
    assert_eq!(
        a.shape(),
        b.shape(),
        "shape mismatch: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    for (i, (x, y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= tol,
            "mismatch at index {i}: {x} vs {y} (diff {diff})"
        );
    }
}
