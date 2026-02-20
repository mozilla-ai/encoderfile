use super::model_type::ModelType;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use tokenizers::{PaddingParams, TruncationParams};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub transform: Option<String>,
    pub lua_libs: Option<LuaLibs>,
}

#[derive(Debug, Serialize, Deserialize, Default, Copy, Clone)]
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

impl TryFrom<Vec<String>> for LuaLibs {
    type Error = anyhow::Error;
    fn try_from(value: Vec<String>) -> Result<LuaLibs> {
        let mut resolved = LuaLibs::default();

        for lib in value {
            match lib.as_str() {
                "coroutine" => resolved.coroutine = true,
                "table" => resolved.table = true,
                "io" => resolved.io = true,
                "os" => resolved.os = true,
                "string" => resolved.string = true,
                "utf8" => resolved.utf8 = true,
                "math" => resolved.math = true,
                "package" => resolved.package = true,
                "debug" => resolved.debug = true,
                other => bail!("Unknown Lua stdlib: {}", other),
            };
        }

        Ok(resolved)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub padding: PaddingParams,
    pub truncation: TruncationParams,
}
