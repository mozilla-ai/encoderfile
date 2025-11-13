use axum::http::StatusCode;
use rmcp::{ErrorData as McpError, model::ErrorCode};
use serde::Serialize;
use serde_json::value::Value::String as SerdeString;
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
            Self::InputError(s) => (StatusCode::UNPROCESSABLE_ENTITY, *s),
            Self::InternalError(s) => (StatusCode::INTERNAL_SERVER_ERROR, *s),
            Self::ConfigError(s) => (StatusCode::INTERNAL_SERVER_ERROR, *s),
        }
    }
}

impl From<ApiError> for McpError {
    fn from(api_error: ApiError) -> McpError {
        match api_error {
            ApiError::InputError(str) => McpError {
                code: ErrorCode::INVALID_REQUEST,
                message: std::borrow::Cow::Borrowed(str),
                data: None,
            },
            ApiError::InternalError(str) => McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: std::borrow::Cow::Borrowed(str),
                data: None,
            },
            ApiError::ConfigError(str) => McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: std::borrow::Cow::Borrowed(str),
                data: None,
            },
        }
    }
}

const ENCODER_DESER_ERROR_MSG: &str = "Encoder response deserialization error";

pub fn to_mcp_error(serde_err: serde_json::Error) -> McpError {
    McpError {
        code: ErrorCode::INVALID_REQUEST,
        message: std::borrow::Cow::Borrowed(ENCODER_DESER_ERROR_MSG),
        data: Some(SerdeString(serde_err.to_string())),
    }
}
