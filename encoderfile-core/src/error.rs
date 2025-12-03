use serde::Serialize;
use thiserror::Error;
use mlua::prelude::LuaError;

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


impl From<RuntimeError> for ApiError {
    fn from(rt_err: RuntimeError) -> ApiError {
        // TODO consider moving strs to strings at some point?
        return ApiError::LuaError(rt_err.to_string());
    }
}


impl From<RuntimeError> for LuaError {
    fn from(rt_err: RuntimeError) -> LuaError {
        // TODO consider moving strs to strings at some point?
        return LuaError::RuntimeError(rt_err.to_string());
    }
}


#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("Internal Error: {0}")]
    InternalError(String)
}
