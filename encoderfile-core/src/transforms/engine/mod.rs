use std::marker::PhantomData;

use crate::error::ApiError;

use super::tensor::Tensor;
use mlua::prelude::*;

mod embedding;
mod sentence_embedding;
mod sequence_classification;
mod token_classification;

macro_rules! transform {
    ($transform_type:ident, $type_name:ident) => {
        pub type $type_name = Transform<$transform_type>;
        pub struct $transform_type;
    };
}

transform!(Embedding, EmbeddingTransform);
transform!(SequenceClassification, SequenceClassificationTransform);
transform!(TokenClassification, TokenClassificationTransform);
transform!(SentenceEmbedding, SentenceEmbeddingTransform);

pub trait Postprocessor: TransformSpec {
    type Input;
    type Output;

    fn postprocess(&self, data: Self::Input) -> Result<Self::Output, ApiError>;
}

pub trait TransformSpec {
    fn has_postprocessor(&self) -> bool;
}

#[derive(Debug)]
pub struct Transform<T> {
    #[allow(dead_code)]
    lua: Lua,
    postprocessor: Option<LuaFunction>,
    _marker: PhantomData<T>,
}

impl<T> Transform<T> {
    fn postprocessor(&self) -> &Option<LuaFunction> {
        &self.postprocessor
    }

    #[tracing::instrument(name = "new_transform", skip_all)]
    pub fn new(transform: Option<&str>) -> Result<Self, ApiError> {
        let lua = new_lua()?;

        lua.load(transform.unwrap_or(""))
            .exec()
            .map_err(|e| ApiError::LuaError(e.to_string()))?;

        let postprocessor = lua
            .globals()
            .get::<Option<LuaFunction>>("Postprocess")
            .map_err(|e| ApiError::LuaError(e.to_string()))?;

        Ok(Self {
            lua,
            postprocessor,
            _marker: PhantomData,
        })
    }
}

impl<T> TransformSpec for Transform<T> {
    fn has_postprocessor(&self) -> bool {
        self.postprocessor.is_some()
    }
}

fn new_lua() -> Result<Lua, ApiError> {
    let lua = Lua::new_with(
        mlua::StdLib::TABLE | mlua::StdLib::STRING | mlua::StdLib::MATH,
        mlua::LuaOptions::default(),
    )
    .map_err(|e| {
        tracing::error!(
            "Failed to create new Lua engine. This should not happen. More details: {:?}",
            e
        );
        ApiError::InternalError("Failed to create new Lua engine")
    })?;

    let globals = lua.globals();
    globals
        .set(
            "Tensor",
            lua.create_function(|lua, value| Tensor::from_lua(value, lua))
                .map_err(|e| {
                    tracing::error!("Failed to create Lua tensor library: More details: {:?}", e);
                    ApiError::InternalError("Failed to create new Lua tensor library")
                })?,
        )
        .map_err(|e| {
            tracing::error!("Failed to create Lua tensor library: More details: {:?}", e);
            ApiError::InternalError("Failed to create new Lua tensor library")
        })?;

    Ok(lua)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_test_lua() -> Lua {
        new_lua().expect("Failed to create new lua")
    }

    #[test]
    fn test_create_tensor() {
        let lua = new_test_lua();
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

    //     #[test]
    //     fn test_bad_output_transform_postprocessing() {
    //         let engine = Transform::new(
    //             r##"
    //         function Postprocess(x)
    //             return 1
    //         end
    //         "##,
    //         )
    //         .unwrap();

    //         let arr = ndarray::Array2::<f32>::from_elem((3, 3), 2.0);

    //         let result = engine.postprocess(arr.clone());

    //         assert!(result.is_err())
    //     }

    //     #[test]
    //     fn test_bad_dimensionality_transform_postprocessing() {
    //         let engine = Transform::new(
    //             r##"
    //         function Postprocess(x)
    //             return x:sum_axis(1)
    //         end
    //         "##,
    //         )
    //         .unwrap();

    //         let arr = ndarray::Array2::<f32>::from_elem((3, 3), 2.0);
    //         let result = engine.postprocess(arr.clone());

    //         assert!(result.is_err());

    //         if let Err(e) = result {
    //             match e {
    //                 ApiError::LuaError(s) => {
    //                     assert!(s.contains("Postprocess function returned tensor of dim"))
    //                 }
    //                 _ => panic!("Didn't return lua error"),
    //             }
    //         }
    //     }
    // }

    #[test]
    fn test_no_unsafe_stdlibs_loaded() {
        let engine = new_test_lua();

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
        let lua = new_lua().expect("Failed to create new Lua");

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
        let lua = new_test_lua();

        // 'require' should not be usable
        let res = lua.load("require('os')").exec();
        assert!(res.is_err());

        // 'package' table should not exist
        let res = lua.load("package").eval::<mlua::Value>();
        assert!(res.unwrap().is_nil())
    }

    #[test]
    fn test_tensor_function_is_only_safe_binding() {
        let lua = new_test_lua();

        // Tensor should exist
        let tensor_res = lua.load("return Tensor").eval::<mlua::Value>();
        assert!(tensor_res.is_ok());

        // But nothing else custom
        let res = lua.load("return DangerousFunction").eval::<mlua::Value>();
        assert!(res.unwrap().is_nil());
    }

    #[test]
    fn test_limited_math_and_string_stdlibs() {
        let lua = new_test_lua();

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
