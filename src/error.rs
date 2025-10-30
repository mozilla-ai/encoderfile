use axum::http::StatusCode;
use thiserror::Error;
use tonic::Status;
use serde::Serialize;

#[derive(Debug, Error, Serialize)]
pub enum ApiError {
    #[error("Invalid Input: {0}")]
    InputError(&'static str),

    #[error("Internal Error: {0}")]
    InternalError(&'static str),

    #[error("Config Error: {0}")]
    ConfigError(&'static str),
}

impl ApiError {
    pub fn to_tonic_status(&self) -> Status {
        match self {
            Self::InputError(s) => Status::invalid_argument(*s),
            Self::InternalError(s) => Status::internal(*s),
            Self::ConfigError(s) => Status::internal(*s),
        }
    }

    pub fn to_axum_status(&self) -> (StatusCode, &'static str) {
        match self {
            Self::InputError(s) => (StatusCode::BAD_REQUEST, *s),
            Self::InternalError(s) => (StatusCode::INTERNAL_SERVER_ERROR, *s),
            Self::ConfigError(s) => (StatusCode::INTERNAL_SERVER_ERROR, *s),
        }
    }
}
