use super::tensor::Tensor;
use mlua::prelude::*;

pub struct TransformEngine {
    lua: Lua,
}

impl TransformEngine {
    pub fn new(postprocessor: &str) -> Result<Self, LuaError> {
        let engine = Self::default();

        engine.lua.load(postprocessor).exec()?;

        Ok(engine)
    }

    pub fn get_function(&self, func: &str) -> Result<LuaFunction, LuaError> {
        self.lua.globals().get(func)
    }

    pub fn exec(&self, chunk: &str) -> Result<(), LuaError> {
        self.lua.load(chunk).exec()?;

        Ok(())
    }

    pub fn eval<T: FromLuaMulti>(&self, chunk: &str) -> Result<T, LuaError> {
        self.lua.load(chunk).eval()
    }

    pub fn postprocess(&self, data: Tensor) -> Result<Tensor, LuaError> {
        let func: LuaFunction = self.lua.globals().get("Postprocess")?;

        func.call(data)
    }
}

impl Default for TransformEngine {
    fn default() -> Self {
        let lua = Lua::new_with(
            mlua::StdLib::TABLE | mlua::StdLib::STRING | mlua::StdLib::MATH,
            mlua::LuaOptions::default(),
        )
        .unwrap();

        let globals = lua.globals();
        globals
            .set(
                "Tensor",
                lua.create_function(|lua, value| Tensor::from_lua(value, lua))
                    .unwrap(),
            )
            .unwrap();

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
        engine
            .exec(
                r#"
            function MyTensor()
                return Tensor({1, 2, 3})
            end
            "#,
            )
            .unwrap();

        let function = engine
            .get_function("MyTensor")
            .expect("Failed to get MyTensor");

        assert!(function.call::<Tensor>(()).is_ok())
    }
}

#[cfg(test)]
mod sandbox_tests {
    use super::*;

    fn setup_engine() -> TransformEngine {
        TransformEngine::default()
    }

    #[test]
    fn test_no_unsafe_stdlibs_loaded() {
        let engine = TransformEngine::default();

        // Should evaluate to nil, not a table or function
        let val: mlua::Value = engine.eval("return os").unwrap();
        assert!(matches!(val, mlua::Value::Nil));

        let val: mlua::Value = engine.eval("return io").unwrap();
        assert!(matches!(val, mlua::Value::Nil));

        let val: mlua::Value = engine.eval("return debug").unwrap();
        assert!(matches!(val, mlua::Value::Nil));
    }

    #[test]
    fn test_cannot_access_environment_or_execute_commands() {
        let engine = TransformEngine::default();

        // `os.execute` shouldn't exist or be callable
        let res =
            engine.eval::<bool>("return type(os) == 'table' and type(os.execute) == 'function'");

        assert!(
            matches!(res, Ok(false) | Err(_)),
            "os.execute should not be callable"
        );
    }

    #[test]
    fn test_no_file_system_access_via_package() {
        let engine = setup_engine();

        // 'require' should not be usable
        let res = engine.exec("require('os')");
        assert!(res.is_err());

        // 'package' table should not exist
        let res = engine.eval::<mlua::Value>("package");
        assert!(res.unwrap().is_nil())
    }

    #[test]
    fn test_tensor_function_is_only_safe_binding() {
        let engine = setup_engine();

        // Tensor should exist
        let tensor_res = engine.eval::<mlua::Value>("return Tensor");
        assert!(tensor_res.is_ok());

        // But nothing else custom
        let res = engine.eval::<mlua::Value>("return DangerousFunction");
        assert!(res.unwrap().is_nil());
    }

    #[test]
    fn test_limited_math_and_string_stdlibs() {
        let engine = setup_engine();

        // math should work
        assert_eq!(engine.eval::<f64>("return math.sqrt(9)").unwrap(), 3.0);

        // string manipulation should work
        assert_eq!(
            engine
                .eval::<String>("return string.upper('sandbox')")
                .unwrap(),
            "SANDBOX"
        );

        // io.open should NOT exist
        assert!(engine.eval::<mlua::Value>("return io.open").is_err());
    }
}
