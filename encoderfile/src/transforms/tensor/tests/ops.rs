use super::{Tensor, add, div, load_env, mul, sub};
use mlua::prelude::*;
use ndarray::{Array0, Array2};

#[test]
fn test_ndim() {
    let lua = load_env();
    let tensor = Tensor(Array2::zeros((3, 3)).into_dyn());

    let ndim = lua.load("return function(x) return x:ndim() end")
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

    let ndim = lua.load("return function(x) return x:ndim() end")
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
