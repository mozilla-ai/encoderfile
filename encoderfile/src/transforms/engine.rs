use crate::error::ApiError;

use super::tensor::Tensor;
use mlua::prelude::*;
use ndarray::Array;

#[derive(Debug)]
pub struct Transform {
    #[allow(dead_code)]
    lua: Lua,
    postprocessor: Option<LuaFunction>,
}

impl Transform {
    #[tracing::instrument(name = "new_transform", skip_all)]
    pub fn new(transform: &str) -> Result<Self, ApiError> {
        let lua = new_lua();

        lua.load(transform)
            .exec()
            .map_err(|e| ApiError::LuaError(e.to_string()))?;

        let postprocessor = lua
            .globals()
            .get::<Option<LuaFunction>>("Postprocess")
            .map_err(|e| ApiError::LuaError(e.to_string()))?;

        Ok(Self { lua, postprocessor })
    }

    #[tracing::instrument(name = "transform_postprocess", skip_all)]
    pub fn postprocess<D: ndarray::Dimension>(
        &self,
        data: Array<f32, D>,
    ) -> Result<Array<f32, D>, ApiError> {
        let func = match &self.postprocessor {
            Some(p) => p,
            None => return Ok(data),
        };

        let data_shape: Vec<usize> = data.shape().to_vec();
        let tensor = Tensor(data.into_dyn());

        let Tensor(result) = func
            .call::<Tensor>(tensor)
            .map_err(|e| ApiError::LuaError(e.to_string()))?;

        if data_shape.as_slice() != result.shape() {
            return Err(ApiError::LuaError(format!(
                "Postprocess function returned tensor of dim {:?}, expected {:?}",
                result.shape(),
                data_shape
            )));
        }

        #[cfg(not(tarpaulin_include))]
        result.into_dimensionality::<D>().map_err(|e| {
            tracing::error!("Failed to cast array into Ix3: {e}");
            ApiError::InternalError(
                "Failed to cast array into correct dim. This is not supposed to happen.",
            )
        })
    }
}

fn new_lua() -> Lua {
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

    lua
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tensor() {
        let lua = new_lua();
        lua.load(
            r#"
            function MyTensor()
                return Tensor({1, 2, 3})
            end
            "#,
        )
        .exec()
        .unwrap();

        let function = lua
            .globals()
            .get::<LuaFunction>("MyTensor")
            .expect("Failed to get MyTensor");

        assert!(function.call::<Tensor>(()).is_ok())
    }

    #[test]
    fn test_no_transform_postprocessing() {
        let engine = Transform::new("").expect("Failed to create Transform");

        let arr = ndarray::Array2::<f32>::from_elem((3, 3), 2.0);

        let result = engine.postprocess(arr.clone()).expect("Failed");

        assert_eq!(arr, result);
    }

    #[test]
    fn test_bad_output_transform_postprocessing() {
        let engine = Transform::new(
            r##"
        function Postprocess(x)
            return 1
        end
        "##,
        )
        .unwrap();

        let arr = ndarray::Array2::<f32>::from_elem((3, 3), 2.0);

        let result = engine.postprocess(arr.clone());

        assert!(result.is_err())
    }

    #[test]
    fn test_bad_dimensionality_transform_postprocessing() {
        let engine = Transform::new(
            r##"
        function Postprocess(x)
            return x:sum_axis(1)
        end
        "##,
        )
        .unwrap();

        let arr = ndarray::Array2::<f32>::from_elem((3, 3), 2.0);
        let result = engine.postprocess(arr.clone());

        assert!(result.is_err());

        if let Err(e) = result {
            match e {
                ApiError::LuaError(s) => {
                    assert!(s.contains("Postprocess function returned tensor of dim"))
                }
                _ => panic!("Didn't return lua error"),
            }
        }
    }
}

#[cfg(test)]
mod sandbox_tests {
    use super::*;

    #[test]
    fn test_no_unsafe_stdlibs_loaded() {
        let engine = new_lua();

        // Should evaluate to nil, not a table or function
        let val: mlua::Value = engine.load("return os").eval().unwrap();
        assert!(matches!(val, mlua::Value::Nil));

        let val: mlua::Value = engine.load("return io").eval().unwrap();
        assert!(matches!(val, mlua::Value::Nil));

        let val: mlua::Value = engine.load("return debug").eval().unwrap();
        assert!(matches!(val, mlua::Value::Nil));
    }

    #[test]
    fn test_cannot_access_environment_or_execute_commands() {
        let lua = new_lua();

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
        let lua = new_lua();

        // 'require' should not be usable
        let res = lua.load("require('os')").exec();
        assert!(res.is_err());

        // 'package' table should not exist
        let res = lua.load("package").eval::<mlua::Value>();
        assert!(res.unwrap().is_nil())
    }

    #[test]
    fn test_tensor_function_is_only_safe_binding() {
        let lua = new_lua();

        // Tensor should exist
        let tensor_res = lua.load("return Tensor").eval::<mlua::Value>();
        assert!(tensor_res.is_ok());

        // But nothing else custom
        let res = lua.load("return DangerousFunction").eval::<mlua::Value>();
        assert!(res.unwrap().is_nil());
    }

    #[test]
    fn test_limited_math_and_string_stdlibs() {
        let lua = new_lua();

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
