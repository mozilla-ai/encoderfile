use std::marker::PhantomData;

use crate::{
    common::{LuaLibs, model_type::{self, ModelTypeSpec}},
    error::ApiError, transforms::DEFAULT_LIBS,
};

use super::tensor::Tensor;
use mlua::prelude::*;

mod embedding;
mod sentence_embedding;
mod sequence_classification;
mod token_classification;

impl From<&LuaLibs> for Vec<mlua::StdLib> {
    fn from(value: &LuaLibs) -> Self {
        let mut libs = Vec::new();
        if value.coroutine {
            libs.push(mlua::StdLib::COROUTINE);
        }
        if value.table {
            libs.push(mlua::StdLib::TABLE);
        }
        if value.io {
            libs.push(mlua::StdLib::IO);
        }
        if value.os {
            libs.push(mlua::StdLib::OS);
        }
        if value.string {
            libs.push(mlua::StdLib::STRING);
        }
        if value.utf8 {
            libs.push(mlua::StdLib::UTF8);
        }
        if value.math {
            libs.push(mlua::StdLib::MATH);
        }
        if value.package {
            libs.push(mlua::StdLib::PACKAGE);
        }
        // luau settings (https://luau.org/), not included right now
        /*
        if value.buffer {
            libs.push(mlua::StdLib::BUFFER);
        }
        if value.vector {
            libs.push(mlua::StdLib::VECTOR);
        }
        */
        // luajit settings (https://luajit.org/), not included right now
        /*
        if value.jit {
            libs.push(mlua::StdLib::JIT);
        }
        if value.ffi {
            libs.push(mlua::StdLib::FFI);
        }
        */
        if value.debug {
            libs.push(mlua::StdLib::DEBUG);
        }
        libs
    }
}

pub fn convert_libs(value: Option<&LuaLibs>) -> Vec<mlua::StdLib>  {
    match value {
        Some(libs) => Vec::from(libs),
        None => DEFAULT_LIBS.to_vec(),
    }   
}

macro_rules! transform {
    ($type_name:ident, $mt:ident) => {
        pub type $type_name = Transform<model_type::$mt>;
    };
}

transform!(EmbeddingTransform, Embedding);
transform!(SequenceClassificationTransform, SequenceClassification);
transform!(TokenClassificationTransform, TokenClassification);
transform!(SentenceEmbeddingTransform, SentenceEmbedding);

pub trait Postprocessor: TransformSpec {
    type Input;
    type Output;

    fn postprocess(&self, data: Self::Input) -> Result<Self::Output, ApiError>;
}

pub trait TransformSpec {
    fn has_postprocessor(&self) -> bool;
}

#[derive(Debug)]
pub struct Transform<T: ModelTypeSpec> {
    #[allow(dead_code)]
    lua: Lua,
    postprocessor: Option<LuaFunction>,
    _marker: PhantomData<T>,
}

impl<T: ModelTypeSpec> Transform<T> {
    fn postprocessor(&self) -> &Option<LuaFunction> {
        &self.postprocessor
    }

    #[tracing::instrument(name = "new_transform", skip_all)]
    pub fn new(libs: Vec<mlua::StdLib>, transform: Option<String>) -> Result<Self, ApiError> {
        let lua = new_lua(libs)?;

        lua.load(transform.unwrap_or("".to_string()))
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

impl<T: ModelTypeSpec> TransformSpec for Transform<T> {
    fn has_postprocessor(&self) -> bool {
        self.postprocessor.is_some()
    }
}

fn new_lua(libs: Vec<mlua::StdLib>) -> Result<Lua, ApiError> {
    let lua = Lua::new_with(
        libs.iter().fold(mlua::StdLib::NONE, |acc, lib| acc | *lib),
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
    use crate::transforms::DEFAULT_LIBS;

    fn new_test_lua() -> Lua {
        new_lua(DEFAULT_LIBS.to_vec()).expect("Failed to create new Lua")
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
        let lua = new_lua(DEFAULT_LIBS.to_vec()).expect("Failed to create new Lua");

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

    #[test]
    fn test_tensor_metatable_preserved() {
        let lua = new_test_lua();

        lua.load(
            r#"
            function mt(t)
                local a = t:layer_norm(1, 1e-5)
                return getmetatable(a) == getmetatable(t)
            end
        "#,
        )
        .exec()
        .unwrap();

        let t = Tensor(ndarray::Array1::<f32>::ones(3).into_dyn());
        let f = lua.globals().get::<LuaFunction>("mt").unwrap();

        let same: bool = f.call(t).unwrap();
        assert!(same, "returned tensor lost its metatable");
    }

    #[test]
    fn test_tensor_return_type() {
        let lua = new_test_lua();

        lua.load(
            r#"
            function ty(t)
                local a = t:layer_norm(1, 1e-5)
                return type(a)
            end
        "#,
        )
        .exec()
        .unwrap();

        let t = Tensor(ndarray::Array1::<f32>::ones(3).into_dyn());
        let f = lua.globals().get::<LuaFunction>("ty").unwrap();

        let ty: String = f.call(t).unwrap();
        assert_eq!(ty, "userdata");
    }

    #[test]
    fn test_tensor_methods_chain_twice() {
        let lua = new_test_lua();

        lua.load(
            r#"
            function chain(t)
                return t
                    :layer_norm(1, 1e-5)
                    :layer_norm(1, 1e-5)
            end
        "#,
        )
        .exec()
        .unwrap();

        let t = Tensor(ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn());

        let f = lua.globals().get::<LuaFunction>("chain").unwrap();

        let out: Tensor = f.call(t).unwrap();

        // shape should be preserved
        assert_eq!(out.0.shape(), &[3]);
    }
}
