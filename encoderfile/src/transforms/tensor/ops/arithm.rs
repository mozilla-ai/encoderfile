use super::Tensor;
use super::properties::is_broadcastable;
use mlua::prelude::*;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn exp(&self) -> Result<Self, LuaError> {
        Ok(Self(self.0.exp()))
    }
}

#[tracing::instrument(skip_all)]
pub fn add(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            if !is_broadcastable(this.shape(), oth.shape()) {
                return Err(LuaError::external(format!(
                    "Shape {:?} not broadcastable to {:?}",
                    this.shape(),
                    oth.shape()
                )));
            }

            this + oth
        }
        LuaValue::Number(n) => this + (n as f32),
        LuaValue::Integer(i) => this + (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

#[tracing::instrument(skip_all)]
pub fn sub(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            if !is_broadcastable(oth.shape(), this.shape()) {
                return Err(LuaError::external(format!(
                    "Shape {:?} not broadcastable to {:?}",
                    this.shape(),
                    oth.shape()
                )));
            }

            this - oth
        }
        LuaValue::Number(n) => this - (n as f32),
        LuaValue::Integer(i) => this - (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

#[tracing::instrument(skip_all)]
pub fn mul(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            if !is_broadcastable(this.shape(), oth.shape()) {
                return Err(LuaError::external(format!(
                    "Shape {:?} not broadcastable to {:?}",
                    this.shape(),
                    oth.shape()
                )));
            }

            this * oth
        }
        LuaValue::Number(n) => this * (n as f32),
        LuaValue::Integer(i) => this * (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

#[tracing::instrument(skip_all)]
pub fn div(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            if !is_broadcastable(oth.shape(), this.shape()) {
                return Err(LuaError::external(format!(
                    "Shape {:?} not broadcastable to {:?}",
                    this.shape(),
                    oth.shape()
                )));
            }

            this / oth
        }
        LuaValue::Number(n) => this / (n as f32),
        LuaValue::Integer(i) => this / (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

#[cfg(test)]
fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor(ndarray::ArrayD::from_shape_vec(shape, data).unwrap())
}

#[cfg(test)]
fn lua_number(n: f64) -> LuaValue {
    LuaValue::Number(n)
}

#[cfg(test)]
fn lua_tensor(t: Tensor, lua: &Lua) -> LuaValue {
    mlua::Value::UserData(lua.create_userdata(t).unwrap())
}

macro_rules! generate_ops_test {
    ($mod_name:ident, $op:tt, $rust_fn:ident, $lua_op:expr) => {
        mod $mod_name {

            #[test]
            fn test_binding() {
                use crate::transforms::tensor::load_env;
                use super::Tensor;
                use super::$rust_fn;
                use ndarray::Array2;
                use mlua::prelude::{LuaValue, LuaFunction};

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
                use crate::transforms::tensor::load_env
;
                use super::Tensor;
                use ndarray::Array2;
                use mlua::prelude::LuaValue;
                use super::$rust_fn;

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
                use super::Tensor;
                use ndarray::Array2;
                use mlua::prelude::LuaValue;
                use super::$rust_fn;

                let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());

                let gold_sum = &arr1.0 $op Array2::<f32>::from_elem((3, 3), 5.0);

                let result = $rust_fn(&arr1, LuaValue::Number(5.0)).unwrap();

                assert_eq!(gold_sum, result.0);
            }

            #[test]
            fn test_integer() {
                use super::Tensor;
                use ndarray::Array2;
                use mlua::prelude::LuaValue;
                use super::$rust_fn;

                let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());

                let gold_sum = &arr1.0 $op Array2::<f32>::from_elem((3, 3), 5.0);

                let result = $rust_fn(&arr1, LuaValue::Integer(5)).unwrap();

                assert_eq!(gold_sum, result.0);
            }

            #[test]
            fn test_bad_dtype() {
                use super::Tensor;
                use ndarray::Array2;
                use mlua::prelude::{LuaValue, LuaError};
                use super::$rust_fn;

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
fn test_exp() {
    use ndarray::Array2;

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
