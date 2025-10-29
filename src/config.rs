use std::sync::OnceLock;

use crate::assets::MODEL_CONFIG_JSON;

use serde::{Deserialize, Serialize};

static MODEL_CONFIG: OnceLock<ModelConfig> = OnceLock::new();

pub fn get_model_config() -> &'static ModelConfig {
    MODEL_CONFIG.get_or_init(
        || match serde_json::from_str::<ModelConfig>(MODEL_CONFIG_JSON) {
            Ok(c) => c,
            Err(e) => panic!("FATAL: Error loading model config: {e:?}"),
        },
    )
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub pad_token_id: u32,
}
