use super::model_type::ModelType;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Config {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub transform: Option<String>,
    pub tokenizer: Option<TokenizerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TokenizerConfig {
    pub pad_strategy: Option<PadStrategyEnum>,
    pub pad_direction: Option<PadDirectionEnum>,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: Option<u32>,
    pub pad_type_id: Option<u32>,
    pub pad_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PadStrategyEnum {
    Longest,
    Fixed(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PadDirectionEnum {
    Left,
    Right,
}
