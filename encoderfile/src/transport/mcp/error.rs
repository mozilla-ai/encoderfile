use crate::error::ApiError;
use rmcp::{ErrorData as McpError, model::ErrorCode};
use serde_json::value::Value::String as SerdeString;

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
            ApiError::LuaError(str) => McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: std::borrow::Cow::Owned(str),
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
