use crate::error::ApiError;

pub trait Inference {
    type Input: serde::de::DeserializeOwned;
    type Output: serde::Serialize;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError>;
}
