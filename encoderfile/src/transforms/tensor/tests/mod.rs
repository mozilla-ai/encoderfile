use super::*;

mod linalg;
mod ops;
mod tensor;

fn load_env() -> Lua {
    let lua = Lua::new();
    lua
}
