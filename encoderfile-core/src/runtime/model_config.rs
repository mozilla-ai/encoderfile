use crate::common::ModelConfig;
use std::sync::{Arc, OnceLock};

static MODEL_CONFIG: OnceLock<Arc<ModelConfig>> = OnceLock::new();

pub fn get_model_config(config_str: &str) -> Arc<ModelConfig> {
    MODEL_CONFIG
        .get_or_init(|| match serde_json::from_str::<ModelConfig>(config_str) {
            Ok(c) => Arc::new(c),
            Err(e) => panic!("FATAL: Error loading model config: {e:?}"),
        })
        .clone()
}
