use super::*;

mod linalg;
mod ops;
mod tensor;

fn load_env() -> Lua {
    Lua::new()
}
