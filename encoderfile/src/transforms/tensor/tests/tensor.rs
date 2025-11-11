use super::*;

#[test]
fn test_from_lua_create_table() {
    let lua = load_env();

    let tbl: LuaTable = lua.load("return {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}").eval().unwrap();

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
