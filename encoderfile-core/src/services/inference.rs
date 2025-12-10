use crate::{common::FromCliInput, error::ApiError};

pub trait Inference {
    type Input: FromCliInput + serde::de::DeserializeOwned + Sync + Send;
    type Output: serde::Serialize + Sync + Send;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError>;
}
