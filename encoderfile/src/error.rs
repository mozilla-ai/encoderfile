use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error, Serialize)]
pub enum ApiError {
    #[error("Invalid Input: {0}")]
    InputError(&'static str),

    #[error("Internal Error: {0}")]
    InternalError(&'static str),

    #[error("Config Error: {0}")]
    ConfigError(&'static str),

    #[error("Lua Error: {0}")]
    LuaError(String),
}
