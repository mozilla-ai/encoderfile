use super::utils::table_to_vec;
use mlua::prelude::*;
use ops::arithm::{add, sub, mul, div};
use ndarray::ArrayD;

mod ops;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor(pub ArrayD<f32>);

impl Tensor {
    pub fn into_inner(self) -> ArrayD<f32> {
        self.0
    }
}

fn load_env() -> Lua {
    Lua::new()
}

impl FromLua for Tensor {
    fn from_lua(value: LuaValue, _lua: &Lua) -> Result<Tensor, LuaError> {
        match value {
            LuaValue::Table(tbl) => {
                let mut shape = Vec::new();

                let vec = table_to_vec(&tbl, &mut shape)?;

                ArrayD::from_shape_vec(shape.as_slice(), vec)
                    .map_err(|e| {
                        LuaError::external(format!("Failed to cast to dimensionality: {e}"))
                    })
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

        methods.add_meta_method(LuaMetaMethod::Len, |_, this, _: ()| Ok(this.len()));

        methods.add_meta_method(LuaMetaMethod::Add, |_, this, other| add(this, other));
        methods.add_meta_method(LuaMetaMethod::Sub, |_, this, other| sub(this, other));
        methods.add_meta_method(LuaMetaMethod::Mul, |_, this, other| mul(this, other));
        methods.add_meta_method(LuaMetaMethod::Div, |_, this, other| div(this, other));

        methods.add_meta_method(LuaMetaMethod::ToString, |_, this, _: ()| {
            Ok(this.0.to_string())
        });

        // tensor ops
        methods.add_method("std", |_, this, ddof| this.std(ddof));
        methods.add_method("mean", |_, this, _: ()| this.mean());
        methods.add_method("ndim", |_, this, _: ()| this.ndim());
        methods.add_method("softmax", |_, this, axis: isize| this.softmax(axis));
        methods.add_method("transpose", |_, this, _: ()| this.transpose());
        methods.add_method("lp_normalize", |_, this, (p, axis)| {
            this.lp_normalize(p, axis)
        });
        methods.add_method("min", |_, this, _: ()| this.min());
        methods.add_method("max", |_, this, _: ()| this.max());
        methods.add_method("exp", |_, this, _: ()| this.exp());
        methods.add_method("sum_axis", |_, this, axis| this.sum_axis(axis));
        methods.add_method("sum", |_, this, _: ()| this.sum());

        methods.add_method("map_axis", |_, this, (axis, func)| {
            this.map_axis(axis, func)
        });
        methods.add_method("fold_axis", |_, this, (axis, acc, func)| {
            this.fold_axis(axis, acc, func)
        });
        methods.add_method("mean_pool", |_, this, mask| this.mean_pool(mask));
        methods.add_method("clamp", |_, this, (min, max)| this.clamp(min, max));
        methods.add_method("layer_norm", |_, this, (axis, eps)| {
            this.layer_norm(axis, eps)
        });
        methods.add_method("truncate_axis", |_, this, (axis, len)| {
            this.truncate_axis(axis, len)
        });
    }
}

#[test]
fn test_from_lua_create_table() {
    let lua = load_env();

    let tbl: LuaTable = lua
        .load("return {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}")
        .eval()
        .unwrap();

    let tensor = Tensor::from_lua(LuaValue::Table(tbl), &lua).expect("Failed to create tensor");

    assert_eq!(tensor.0.ndim(), 2);
    assert_eq!(tensor.0.shape(), [3, 3]);
}

#[test]
fn test_from_lua_empty_table() {
    let lua = load_env();

    let tbl: LuaTable = lua.load("return {}").eval().unwrap();

    let Tensor(tensor) = Tensor::from_lua(LuaValue::Table(tbl), &lua).unwrap();

    assert!(tensor.is_empty());
    assert_eq!(tensor.ndim(), 1);
}

#[test]
fn test_from_lua_ragged() {
    let lua = load_env();

    let tbl: LuaTable = lua
        .load("return {{1, 1, 1}, {1, 1, 1}, {1, 1}}")
        .eval()
        .unwrap();

    let tensor = Tensor::from_lua(LuaValue::Table(tbl), &lua);

    assert!(tensor.is_err());
}

#[test]
fn test_from_lua_bad_type() {
    let lua = load_env();

    let tbl: LuaString = lua.load("return \"i am not a table\"").eval().unwrap();

    let tensor = Tensor::from_lua(LuaValue::String(tbl), &lua);

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
    use ndarray::Array2;

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
    use ndarray::Array2;

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
    use ndarray::Array2;

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

