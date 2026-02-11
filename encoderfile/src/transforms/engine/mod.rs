use std::marker::PhantomData;

use crate::{
    common::{
        LuaLibs,
        model_type::{self, ModelTypeSpec},
    },
    error::ApiError,
    transforms::DEFAULT_LIBS,
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

pub fn convert_libs(value: Option<&LuaLibs>) -> Vec<mlua::StdLib> {
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

    enum TestLibItem {
        Coroutine,
        Io,
        Utf8,
        Os,
        Package,
        #[allow(dead_code)]
        Debug,
    }

    impl TestLibItem {
        pub fn test_data(self) -> (String, mlua::StdLib) {
            match self {
                TestLibItem::Coroutine => (
                    r#"
                    function MyCoroutine()
                        return Tensor({1, 2, 3})
                    end
                    function MyTest()
                        local mycor = coroutine.create(MyCoroutine)
                        local _, tensor = coroutine.resume(mycor)
                        return tensor
                    end
                "#
                    .to_string(),
                    mlua::StdLib::COROUTINE,
                ),
                TestLibItem::Io => (
                    r#"
                    function MyTest()
                        local res = Tensor({1, 2, 3})
                        io.stderr:write("This is a test of the IO library\n")
                        return res
                    end
                "#
                    .to_string(),
                    mlua::StdLib::IO,
                ),
                TestLibItem::Utf8 => (
                    r#"
                    function MyTest()
                        local fp_values = {}
                        for point in utf8.codes("hello") do
                            table.insert(fp_values, point)
                        end
                        return Tensor(fp_values)
                    end
                "#
                    .to_string(),
                    mlua::StdLib::UTF8,
                ),
                TestLibItem::Os => (
                    r#"
                    function MyTest()
                        local t = os.time()
                        return Tensor({1, 2, 3})
                    end
                "#
                    .to_string(),
                    mlua::StdLib::OS,
                ),
                TestLibItem::Package => (
                    r#"
                    function MyTest()
                        p = package.path
                        return Tensor({1, 2, 3})
                    end
                "#
                    .to_string(),
                    mlua::StdLib::PACKAGE,
                ),
                TestLibItem::Debug => (
                    r#"
                    function MyTest()
                        local info = debug.getinfo(1, "n")
                        return Tensor({info.currentline})
                    end
                "#
                    .to_string(),
                    mlua::StdLib::DEBUG,
                ),
            }
        }
    }

    #[test]
    fn test_convert_default_lua_libs() {
        let libs = LuaLibs::default();
        let stdlibs: Vec<mlua::StdLib> = Vec::from(&libs);
        assert!(stdlibs.is_empty());
    }

    #[test]
    fn test_convert_no_lua_libs() {
        let maybe_libs = None;
        let stdlibs: Vec<mlua::StdLib> = convert_libs(maybe_libs);
        assert!(stdlibs.contains(&mlua::StdLib::TABLE));
        assert!(stdlibs.contains(&mlua::StdLib::STRING));
        assert!(stdlibs.contains(&mlua::StdLib::MATH));
    }

    #[test]
    fn test_convert_some_lua_libs() {
        let maybe_libs = Some(&LuaLibs {
            coroutine: true,
            table: false,
            io: true,
            os: false,
            string: true,
            utf8: false,
            math: true,
            package: false,
            debug: true,
        });
        let stdlibs: Vec<mlua::StdLib> = convert_libs(maybe_libs);
        assert!(stdlibs.contains(&mlua::StdLib::COROUTINE));
        assert!(stdlibs.contains(&mlua::StdLib::IO));
        assert!(stdlibs.contains(&mlua::StdLib::STRING));
        assert!(stdlibs.contains(&mlua::StdLib::MATH));
        assert!(stdlibs.contains(&mlua::StdLib::DEBUG));
        assert!(!stdlibs.contains(&mlua::StdLib::TABLE));
        assert!(!stdlibs.contains(&mlua::StdLib::OS));
        assert!(!stdlibs.contains(&mlua::StdLib::UTF8));
        assert!(!stdlibs.contains(&mlua::StdLib::PACKAGE));
    }

    fn test_lualib_any_ok((chunk, lib): (String, mlua::StdLib)) {
        let mut lualibs = DEFAULT_LIBS.to_vec();
        lualibs.push(lib);
        let lua = new_lua(lualibs).expect("Failed to create new Lua");
        lua.load(chunk).exec().unwrap();

        let function = lua
            .globals()
            .get::<LuaFunction>("MyTest")
            .expect("Failed to get MyTest");
        let res = function.call::<Tensor>(());
        assert!(
            res.is_ok(),
            "Failed to execute function using library {:?}: {:?}",
            lib,
            res.err()
        );
    }

    fn test_lualib_any_fails((chunk, lib): (String, mlua::StdLib)) {
        let lua = new_test_lua();
        lua.load(chunk).exec().unwrap();

        let function = lua
            .globals()
            .get::<LuaFunction>("MyTest")
            .expect("Failed to get MyTest");
        let res = function.call::<Tensor>(());
        assert!(
            res.is_err(),
            "Function should have failed when using library {:?}, but got result: {:?}",
            lib,
            res.ok()
        );
    }

    #[test]
    fn test_lualib_coroutine_ok() {
        test_lualib_any_ok(TestLibItem::Coroutine.test_data());
    }

    #[test]
    fn test_lualib_coroutine_fails() {
        test_lualib_any_fails(TestLibItem::Coroutine.test_data());
    }

    #[test]
    fn test_lualib_io_ok() {
        test_lualib_any_ok(TestLibItem::Io.test_data());
    }

    #[test]
    fn test_lualib_io_fails() {
        test_lualib_any_fails(TestLibItem::Io.test_data());
    }

    #[test]
    fn test_lualib_utf8_ok() {
        test_lualib_any_ok(TestLibItem::Utf8.test_data());
    }

    #[test]
    fn test_lualib_utf8_fails() {
        test_lualib_any_fails(TestLibItem::Utf8.test_data());
    }

    #[test]
    fn test_lualib_os_ok() {
        test_lualib_any_ok(TestLibItem::Os.test_data());
    }

    #[test]
    fn test_lualib_os_fails() {
        test_lualib_any_fails(TestLibItem::Os.test_data());
    }

    #[test]
    fn test_lualib_package_ok() {
        test_lualib_any_ok(TestLibItem::Package.test_data());
    }

    #[test]
    fn test_lualib_package_fails() {
        test_lualib_any_fails(TestLibItem::Package.test_data());
    }

    // TODO: check lua engine init with the debug lib enabled;
    // tests currently fail here
    /*
    #[test]
    fn test_lualib_debug_ok() {
        test_lualib_any_ok(TestLibItem::Debug.test_data());
    }

    #[test]
    fn test_lualib_debug_fails() {
        test_lualib_any_fails(TestLibItem::Debug.test_data());
    }
    */
}
