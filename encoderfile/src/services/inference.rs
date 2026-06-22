use crate::{common::FromCliInput, error::ApiError, services::Metadata};

// FIXME enforce the openapi schema later on
pub trait Inference: Metadata {
    type Input: FromCliInput + serde::de::DeserializeOwned + Sync + Send;
    type Output: serde::Serialize + Sync + Send;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError>;
}
