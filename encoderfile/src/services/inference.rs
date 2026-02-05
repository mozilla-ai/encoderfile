use crate::{common::FromCliInput, error::ApiError, services::Metadata};

pub trait Inference: Metadata {
    type Input: FromCliInput + serde::de::DeserializeOwned + Sync + Send + utoipa::ToSchema;
    type Output: serde::Serialize + Sync + Send + utoipa::ToSchema;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError>;
}
