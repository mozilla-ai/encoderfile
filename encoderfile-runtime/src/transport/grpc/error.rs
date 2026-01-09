use tonic::Status;

pub trait ToTonicStatus {
    fn to_tonic_status(&self) -> Status;
}

impl ToTonicStatus for encoderfile_core::error::ApiError {
    fn to_tonic_status(&self) -> Status {
        match self {
            Self::InputError(s) => Status::invalid_argument(*s),
            Self::InternalError(s) => Status::internal(*s),
            Self::ConfigError(s) => Status::internal(*s),
            Self::LuaError(s) => Status::internal(s),
        }
    }
}
