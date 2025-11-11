use super::*;

mod linalg;
mod ops;
mod tensor;

const UTILS: &str = include_str!("../../utils.lua");

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
