use mlua::prelude::*;
use super::tensor::Tensor;

pub struct TransformEngine {
    lua: Lua
}

impl TransformEngine {
    pub fn postprocess(&self, data: Tensor) -> Result<Tensor, LuaError> {
        self.lua
            .globals()
            .get::<LuaFunction>("Postprocess")?
            .call(data)
    }
}

impl Default for TransformEngine {
    fn default() -> Self {
        let lua = Lua::new();
        let globals = lua.globals();
        globals.set("Tensor", lua.create_function(|lua, value| {
            Tensor::from_lua(value, lua)
            }).unwrap()).unwrap();
        Self { lua }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_engine_default() {
        let _engine = TransformEngine::default();
    }

    #[test]
    fn test_create_tensor() {
        let engine = TransformEngine::default();
        engine.lua
            .load(r#"
            function MyTensor()
                return Tensor({1, 2, 3})
            end
            "#)
            .exec()
            .unwrap();

        let function: LuaFunction = engine
            .lua
            .globals()
            .get("MyTensor")
            .unwrap();

        assert!(function.call::<Tensor>(()).is_ok())
    }
}
