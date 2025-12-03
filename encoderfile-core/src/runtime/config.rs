use crate::common::{ModelType, ModelConfig};
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
