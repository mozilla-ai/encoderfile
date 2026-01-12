use axum::http::StatusCode;
use std::borrow::Cow;

impl crate::error::ApiError {
    pub fn to_axum_status(&self) -> (StatusCode, Cow<'static, str>) {
        match self {
            Self::InputError(s) => (StatusCode::UNPROCESSABLE_ENTITY, Cow::Borrowed(*s)),
            Self::InternalError(s) => (StatusCode::INTERNAL_SERVER_ERROR, Cow::Borrowed(*s)),
            Self::ConfigError(s) => (StatusCode::INTERNAL_SERVER_ERROR, Cow::Borrowed(*s)),
            Self::LuaError(s) => (StatusCode::INTERNAL_SERVER_ERROR, Cow::Owned(s.to_string())),
        }
    }
}
