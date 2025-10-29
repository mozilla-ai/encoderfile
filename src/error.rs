use thiserror::Error;
use tonic::Status;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Invalid Input: {0}")]
    InputError(&'static str),

    #[error("Internal Error: {0}")]
    InternalError(&'static str)
}

impl Into<Status> for ApiError {
    fn into(self) -> Status {
        match self {
            Self::InputError(s) => Status::invalid_argument(s),
            Self::InternalError(s) => Status::internal(s),
        }
    }
}
