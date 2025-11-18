use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::common::ModelType;
use std::sync::{Arc, OnceLock};

static MODEL_TYPE: OnceLock<ModelType> = OnceLock::new();
static MODEL_CONFIG: OnceLock<Arc<ModelConfig>> = OnceLock::new();

pub fn get_model_config(config_str: &str) -> Arc<ModelConfig> {
    MODEL_CONFIG
        .get_or_init(|| match serde_json::from_str::<ModelConfig>(config_str) {
            Ok(c) => Arc::new(c),
            Err(e) => panic!("FATAL: Error loading model config: {e:?}"),
        })
        .clone()
}

pub fn get_model_type(model_type: &str) -> ModelType {
    MODEL_TYPE
        .get_or_init(|| match model_type {
            "embedding" => ModelType::Embedding,
            "sequence_classification" => ModelType::SequenceClassification,
            "token_classification" => ModelType::TokenClassification,
            "sentence_embedding" => ModelType::SentenceEmbedding,
            other => panic!("Invalid model type: {other}"),
        })
        .clone()
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
        self.label2id.as_ref()?.get(label).copied()
    }
}
