use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use crate::assets::{MODEL_CONFIG_JSON, MODEL_TYPE_STR};

use serde::{Deserialize, Serialize};

static MODEL_CONFIG: OnceLock<Arc<ModelConfig>> = OnceLock::new();
static MODEL_TYPE: OnceLock<ModelType> = OnceLock::new();

pub fn get_model_config() -> Arc<ModelConfig> {
    MODEL_CONFIG
        .get_or_init(
            || match serde_json::from_str::<ModelConfig>(MODEL_CONFIG_JSON) {
                Ok(c) => Arc::new(c),
                Err(e) => panic!("FATAL: Error loading model config: {e:?}"),
            },
        )
        .clone()
}

pub fn get_model_type() -> ModelType {
    MODEL_TYPE
        .get_or_init(|| match MODEL_TYPE_STR {
            "embedding" => ModelType::Embedding,
            "sequence_classification" => ModelType::SequenceClassification,
            "token_classification" => ModelType::TokenClassification,
            other => panic!("Invalid model type: {}", other),
        })
        .clone()
}

#[derive(Debug, Clone)]
pub enum ModelType {
    Embedding,
    SequenceClassification,
    TokenClassification,
}

impl From<ModelType> for crate::generated::encoderfile::ModelType {
    fn from(val: ModelType) -> Self {
        match val {
            ModelType::Embedding => Self::Embedding,
            ModelType::SequenceClassification => Self::SequenceClassification,
            ModelType::TokenClassification => Self::TokenClassification,
        }
    }
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
