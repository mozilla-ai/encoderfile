use crate::error::ApiError;

pub trait Inference {
    type Input;
    type Output;

    fn run(&self, input: Self::Input) -> Result<Self::Output, ApiError>;
}
