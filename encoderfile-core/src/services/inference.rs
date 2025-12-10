use crate::error::ApiError;

pub trait Inference {
    type Input: serde::de::DeserializeOwned + Sync + Send;
    type Output: serde::Serialize + Sync + Send;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError>;
}
