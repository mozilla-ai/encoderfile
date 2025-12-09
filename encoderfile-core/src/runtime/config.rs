use crate::common::{ModelConfig, ModelTypeEnum};
use std::sync::{Arc, OnceLock};

static MODEL_TYPE: OnceLock<ModelTypeEnum> = OnceLock::new();
static MODEL_CONFIG: OnceLock<Arc<ModelConfig>> = OnceLock::new();

pub fn get_model_config(config_str: &str) -> Arc<ModelConfig> {
    MODEL_CONFIG
        .get_or_init(|| match serde_json::from_str::<ModelConfig>(config_str) {
            Ok(c) => Arc::new(c),
            Err(e) => panic!("FATAL: Error loading model config: {e:?}"),
        })
        .clone()
}

pub fn get_model_type(model_type: &str) -> ModelTypeEnum {
    MODEL_TYPE
        .get_or_init(|| match model_type {
            "embedding" => ModelTypeEnum::Embedding,
            "sequence_classification" => ModelTypeEnum::SequenceClassification,
            "token_classification" => ModelTypeEnum::TokenClassification,
            "sentence_embedding" => ModelTypeEnum::SentenceEmbedding,
            other => panic!("Invalid model type: {other}"),
        })
        .clone()
}
