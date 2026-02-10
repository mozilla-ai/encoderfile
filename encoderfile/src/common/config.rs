use super::model_type::ModelType;
use serde::{Deserialize, Serialize};
use tokenizers::PaddingParams;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub transform: Option<String>,
    pub lua_libs: Option<LuaLibs>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LuaLibs {
    pub coroutine: bool,
    pub table: bool,
    pub io: bool,
    pub os: bool,
    pub string: bool,
    pub utf8: bool,
    // Check if / how this is supported in lua54
    // pub bit: bool,
    pub math: bool,
    pub package: bool,
    // luau
    // pub buffer: bool,
    // pub vector: bool,
    // luajit
    // pub jit: bool,
    // pub ffi: bool,
    pub debug: bool,
}

impl Default for LuaLibs {
    fn default() -> Self {
        LuaLibs {
            coroutine: false,
            table: false,
            io: false,
            os: false,
            string: false,
            utf8: false,
            math: false,
            package: false,
            debug: false,
        }
    }
}   

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub padding: PaddingParams,
}
