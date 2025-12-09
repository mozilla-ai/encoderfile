use crate::error::ApiError;

pub trait Inference {
    type Input;
    type Output;

    fn run(&self) -> Result<Self::Output, ApiError>;
}
