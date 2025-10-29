use std::{collections::HashMap, sync::OnceLock};

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
    pub id2label: Option<HashMap<u32, String>>,
    pub label2id: Option<HashMap<String, u32>>,
}

impl ModelConfig {
    pub fn id2label(&self, id: u32) -> Option<&str> {
        self.id2label.as_ref()?.get(&id).map(|s| s.as_str())
    }

    pub fn label2id(&self, label: &str) -> Option<u32> {
        self.label2id.as_ref()?.get(label).map(|i| *i)
    }
}
