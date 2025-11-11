use super::utils::table_to_vec;
use mlua::prelude::*;
use ndarray::{ArrayD, Axis};
use ort::tensor::ArrayExtensions;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor(ArrayD<f32>);

impl FromLua for Tensor {
    fn from_lua(value: LuaValue, _lua: &Lua) -> Result<Self, LuaError> {
        match value {
            LuaValue::Table(tbl) => {
                let mut shape = Vec::new();

                let vec = table_to_vec(&tbl, &mut shape)?;

                ArrayD::from_shape_vec(shape.as_slice(), vec)
                    .map(Self)
                    .map_err(|e| LuaError::external(format!("Shape error: {e}")))
            }
            LuaValue::UserData(data) => data.borrow::<Tensor>().map(|i| i.to_owned()),
            _ => Err(LuaError::external(
                format!("Unknown type: {}", value.type_name()).as_str(),
            )),
        }
    }
}

impl LuaUserData for Tensor {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        // syntactic sugar
        methods.add_meta_method(LuaMetaMethod::Eq, |_, this, other: Tensor| {
            Ok(this.0 == other.0)
        });

        methods.add_meta_method(LuaMetaMethod::Add, |_, this, other| add(this, other));
        methods.add_meta_method(LuaMetaMethod::Sub, |_, this, other| sub(this, other));
        methods.add_meta_method(LuaMetaMethod::Mul, |_, this, other| mul(this, other));
        methods.add_meta_method(LuaMetaMethod::Div, |_, this, other| div(this, other));

        // tensor ops
        methods.add_method("softmax", |_, this, axis: isize| this.softmax(axis));
        methods.add_method("lp_norm", |_, this, p: f32| this.lp_norm(p));
    }
}

impl Tensor {
    fn softmax(&self, axis: isize) -> Result<Self, LuaError> {
        if axis <= 0 {
            return Err(LuaError::external("Axis must be >= 1."));
        }

        let axis_index = (axis - 1) as usize;

        if axis_index >= self.0.ndim() {
            return Err(LuaError::external("Axis out of range."));
        }

        let res = self.0.softmax(Axis(axis_index));
        Ok(Self(res))
    }

    fn lp_norm(&self, p: f32) -> Result<f32, LuaError> {
        if p == 0.0 {
            return Err(LuaError::external("P has to be larger than 0."));
        }

        if p.is_infinite() {
            return Ok(self.0.iter().map(|v| v.abs()).fold(0.0, f32::max));
        }

        Ok(self
            .0
            .iter()
            .map(|v| v.abs().powf(p))
            .sum::<f32>()
            .powf(1.0 / p))
    }
}

fn add(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            this + oth
        }
        LuaValue::Number(n) => this + (n as f32),
        LuaValue::Integer(i) => this + (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

fn sub(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            this - oth
        }
        LuaValue::Number(n) => this - (n as f32),
        LuaValue::Integer(i) => this - (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

fn mul(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            this * oth
        }
        LuaValue::Number(n) => this * (n as f32),
        LuaValue::Integer(i) => this * (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

fn div(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            this / oth
        }
        LuaValue::Number(n) => this / (n as f32),
        LuaValue::Integer(i) => this / (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    const UTILS: &str = include_str!("utils.lua");

    fn load_env() -> Lua {
        let lua = Lua::new();
        lua.load(UTILS).eval::<()>().expect("Failed to load utils");
        lua
    }

    fn get_function(lua: &Lua, name: &str) -> LuaFunction {
        lua.globals()
            .get::<LuaFunction>(name)
            .expect("Failed to find function {name}")
    }

    #[test]
    fn test_from_lua_create_table() {
        let lua = load_env();

        let tbl: LuaTable = get_function(&lua, "CreateGoodTable").call(()).unwrap();

        let tensor = Tensor::from_lua(LuaValue::Table(tbl), &lua).expect("Failed to create tensor");

        assert_eq!(tensor.0.ndim(), 2);
        assert_eq!(tensor.0.shape(), [3, 3]);
    }

    #[test]
    fn test_from_lua_ragged() {
        let lua = load_env();

        let tbl: LuaTable = get_function(&lua, "CreateRaggedTable").call(()).unwrap();

        let tensor = Tensor::from_lua(LuaValue::Table(tbl), &lua);

        assert!(tensor.is_err());
    }

    #[test]
    fn test_from_lua_bad_type() {
        let lua = load_env();

        let tbl: LuaTable = get_function(&lua, "CreateStringTable").call(()).unwrap();

        let tensor = Tensor::from_lua(LuaValue::Table(tbl), &lua);

        assert!(tensor.is_err());
    }

    #[test]
    fn test_from_lua_bad_type_err() {
        let lua = load_env();

        let val = LuaValue::Boolean(false);

        let tensor = Tensor::from_lua(val, &lua);

        assert!(tensor.is_err());
    }

    #[test]
    fn test_eq_simple() {
        let lua = load_env();

        let arr1 = Tensor(Array2::<f32>::ones((3, 3)).into_dyn());
        let arr2 = arr1.clone();

        assert!(arr1 == arr2);

        let result: bool = get_function(&lua, "TestEq")
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

        let result: bool = get_function(&lua, "TestEq")
            .call((arr1, arr2))
            .expect("Failed to evaluate");

        assert!(!result);
    }

    macro_rules! generate_ops_test {
        ($mod_name:ident, $op:tt, $rust_fn:ident, $lua_name:expr) => {
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

                    let result: Tensor = lua.globals()
                        .get::<LuaFunction>($lua_name)
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
        test_addition, +, add, "TestAddition"
    );

    generate_ops_test!(
        test_subtraction, -, sub, "TestSubtraction"
    );

    generate_ops_test!(
        test_multiplication, *, mul, "TestMultiplication"
    );

    generate_ops_test!(
        test_division, /, div, "TestDivision"
    );

    #[test]
    fn test_softmax() {
        let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0], [4.0, 5.0],].into_dyn();

        let ts = Tensor(arr.clone());

        assert_eq!(arr.softmax(Axis(0)), ts.softmax(1).unwrap().0);
        assert_eq!(arr.softmax(Axis(1)), ts.softmax(2).unwrap().0);

        assert_ne!(arr.softmax(Axis(1)), ts.softmax(1).unwrap().0);
    }

    #[test]
    fn test_softmax_fail() {
        let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0], [4.0, 5.0],].into_dyn();

        let ts = Tensor(arr.clone());

        assert!(ts.softmax(-1).is_err());
        assert!(ts.softmax(0).is_err());
        assert!(ts.softmax(3).is_err());
    }

    #[test]
    fn test_l1_norm() {
        // L1 norm = sum of absolute values
        let arr: ArrayD<f32> =
            ndarray::array![[1.0, -2.0, 3.0], [4.0, -5.0, 6.0], [7.0, 8.0, -9.0]].into_dyn();

        let tensor = Tensor(arr);
        let norm = tensor.lp_norm(1.0).unwrap();

        let expected = (1..=9).map(|v| v as f32).sum::<f32>();
        assert!(
            (norm - expected).abs() < 1e-6,
            "Expected {expected}, got {norm}"
        );
    }

    #[test]
    fn test_l2_norm() {
        // L2 norm = sqrt(sum(x^2))
        let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0, 2.0]].into_dyn();

        let tensor = Tensor(arr);
        let norm = tensor.lp_norm(2.0).unwrap();

        let expected = (1.0f32.powi(2) + 2.0f32.powi(2) + 2.0f32.powi(2)).sqrt();
        assert!(
            (norm - expected).abs() < 1e-6,
            "Expected {expected}, got {norm}"
        );
    }

    #[test]
    fn test_lp_norm_fractional_p() {
        // fractional p still valid (e.g. p = 0.5)
        let arr: ArrayD<f32> = ndarray::array![[1.0, 4.0]].into_dyn();
        let tensor = Tensor(arr);
        let norm = tensor.lp_norm(0.5).unwrap();

        let expected = (1.0f32.abs().powf(0.5) + 4.0f32.abs().powf(0.5)).powf(1.0 / 0.5);
        assert!(
            (norm - expected).abs() < 1e-6,
            "Expected {expected}, got {norm}"
        );
    }

    #[test]
    fn test_lp_norm_zero_p_fails() {
        // p == 0.0 should error
        let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0, 3.0]].into_dyn();
        let tensor = Tensor(arr);
        let result = tensor.lp_norm(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lp_norm_empty_tensor() {
        // empty tensors should return 0 (sum over empty iterator)
        let arr: ArrayD<f32> = ArrayD::<f32>::zeros(ndarray::IxDyn(&[0]));
        let tensor = Tensor(arr);
        let norm = tensor.lp_norm(2.0).unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_linf_norm() {
        // L∞ norm = max(|x_i|)
        let arr: ArrayD<f32> = ndarray::array![[1.0, -4.0, 2.0], [3.0, 9.0, -7.0]].into_dyn();

        let tensor = Tensor(arr);
        let norm = tensor.lp_norm(f32::INFINITY).unwrap();

        let expected = 9.0;
        assert!(
            (norm - expected).abs() < 1e-6,
            "Expected {expected}, got {norm}"
        );
    }
}
