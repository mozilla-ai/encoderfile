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
    pub fn postprocess(
        &self,
        data: Tensor,
    ) -> Result<Tensor, LuaError> {
        let func: LuaFunction = self.lua.globals().get("Postprocess")?;

        func.call(data)

        // let mut results = Vec::with_capacity(data.len());

        // for pred in data.0.axis_iter(Axis(0)) {
        //     let result: Tensor = func.call((Tensor(pred.to_owned()), metadata.clone()))?;
        //     results.push(result.0);
        // }

        // let result_views = results.iter().map(|i| i.view()).collect::<Vec<_>>();

        // Ok(Tensor(
        //     ndarray::stack(Axis(0), result_views.as_slice()).unwrap(),
        // ))
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
            .lua
            .load(
                r#"
            function MyTensor()
                return Tensor({1, 2, 3})
            end
            "#,
            )
            .exec()
            .unwrap();

        let function: LuaFunction = engine.lua.globals().get("MyTensor").unwrap();

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
        let lua = &engine.lua;

        // Should evaluate to nil, not a table or function
        let val: mlua::Value = lua.load("return os").eval().unwrap();
        assert!(matches!(val, mlua::Value::Nil));

        let val: mlua::Value = lua.load("return io").eval().unwrap();
        assert!(matches!(val, mlua::Value::Nil));

        let val: mlua::Value = lua.load("return debug").eval().unwrap();
        assert!(matches!(val, mlua::Value::Nil));
    }

    #[test]
    fn test_cannot_access_environment_or_execute_commands() {
        let engine = TransformEngine::default();
        let lua = &engine.lua;

        // `os.execute` shouldn't exist or be callable
        let res = lua
            .load("return type(os) == 'table' and type(os.execute) == 'function'")
            .eval::<bool>();
        assert!(
            matches!(res, Ok(false) | Err(_)),
            "os.execute should not be callable"
        );
    }

    #[test]
    fn test_no_file_system_access_via_package() {
        let engine = setup_engine();
        let lua = &engine.lua;

        // 'require' should not be usable
        let res = lua.load("require('os')").exec();
        assert!(res.is_err());

        // 'package' table should not exist
        let res = lua.load("package").eval::<mlua::Value>();
        assert!(res.unwrap().is_nil())
    }

    #[test]
    fn test_tensor_function_is_only_safe_binding() {
        let engine = setup_engine();
        let lua = &engine.lua;

        // Tensor should exist
        let tensor_res = lua.load("return Tensor").eval::<mlua::Value>();
        assert!(tensor_res.is_ok());

        // But nothing else custom
        let res = lua.load("return DangerousFunction").eval::<mlua::Value>();
        assert!(res.unwrap().is_nil());
    }

    #[test]
    fn test_limited_math_and_string_stdlibs() {
        let engine = setup_engine();
        let lua = &engine.lua;

        // math should work
        assert_eq!(lua.load("return math.sqrt(9)").eval::<f64>().unwrap(), 3.0);

        // string manipulation should work
        assert_eq!(
            lua.load("return string.upper('sandbox')")
                .eval::<String>()
                .unwrap(),
            "SANDBOX"
        );

        // io.open should NOT exist
        assert!(lua.load("return io.open").eval::<mlua::Value>().is_err());
    }
}
