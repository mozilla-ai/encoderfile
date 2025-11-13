use std::borrow::Cow;

use axum::http::StatusCode;
use serde::Serialize;
use thiserror::Error;
use tonic::Status;

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

impl ApiError {
    pub fn to_tonic_status(&self) -> Status {
        match self {
            Self::InputError(s) => Status::invalid_argument(*s),
            Self::InternalError(s) => Status::internal(*s),
            Self::ConfigError(s) => Status::internal(*s),
            Self::LuaError(s) => Status::internal(s),
        }
    }

    pub fn to_axum_status(&self) -> (StatusCode, Cow<'static, str>) {
        match self {
            Self::InputError(s) => (StatusCode::UNPROCESSABLE_ENTITY, Cow::Borrowed(*s)),
            Self::InternalError(s) => (StatusCode::INTERNAL_SERVER_ERROR, Cow::Borrowed(*s)),
            Self::ConfigError(s) => (StatusCode::INTERNAL_SERVER_ERROR, Cow::Borrowed(*s)),
            Self::LuaError(s) => (StatusCode::INTERNAL_SERVER_ERROR, Cow::Owned(s.to_string())),
        }
    }
}
