use encoderfile_core::error::ApiError;
use rmcp::{ErrorData as McpError, model::ErrorCode};
use serde_json::value::Value::String as SerdeString;

pub trait ToMcpError {
    fn to_mcp_error(&self) -> McpError;
}

impl ToMcpError for ApiError {
    fn to_mcp_error(&self) -> McpError {
        match self {
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
            ApiError::LuaError(str) => McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: std::borrow::Cow::Owned(str.clone()),
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
