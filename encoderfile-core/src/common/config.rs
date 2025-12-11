use super::{
    model_type::ModelType,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Config {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub transform: Option<String>,
}
