use super::model_type::ModelType;
use serde::{Deserialize, Serialize};
use tokenizers::{PaddingParams, TruncationParams};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub transform: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub padding: PaddingParams,
    pub truncation: TruncationParams,
}
