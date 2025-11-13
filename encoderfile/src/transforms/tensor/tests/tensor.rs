use super::*;
use ndarray::IxDyn;

#[test]
fn test_from_lua_create_table() {
    let lua = load_env();

    let tbl: LuaTable = lua
        .load("return {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}")
        .eval()
        .unwrap();

    let tensor = Tensor::<IxDyn>::from_lua(LuaValue::Table(tbl), &lua).expect("Failed to create tensor");

    assert_eq!(tensor.0.ndim(), 2);
    assert_eq!(tensor.0.shape(), [3, 3]);
}

#[test]
fn test_from_lua_empty_table() {
    let lua = load_env();

    let tbl: LuaTable = lua.load("return {}").eval().unwrap();

    let Tensor(tensor) = Tensor::<IxDyn>::from_lua(LuaValue::Table(tbl), &lua).unwrap();

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

    let tensor = Tensor::<IxDyn>::from_lua(LuaValue::Table(tbl), &lua);

    assert!(tensor.is_err());
}

#[test]
fn test_from_lua_bad_type() {
    let lua = load_env();

    let tbl: LuaString = lua.load("return \"i am not a table\"").eval().unwrap();

    let tensor = Tensor::<IxDyn>::from_lua(LuaValue::String(tbl), &lua);

    assert!(tensor.is_err());
}

#[test]
fn test_from_lua_bad_type_err() {
    let lua = load_env();

    let val = LuaValue::Boolean(false);

    let tensor = Tensor::<IxDyn>::from_lua(val, &lua);

    assert!(tensor.is_err());
}
