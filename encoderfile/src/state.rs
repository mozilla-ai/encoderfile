use std::sync::Arc;

use ort::session::Session;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use crate::{
    assets::get_model_id,
    config::{ModelConfig, ModelType, get_model_config, get_model_type},
    model::get_model,
    tokenizer::get_tokenizer,
};

#[derive(Debug, Clone)]
pub struct AppState {
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: Arc<ModelConfig>,
    pub model_type: ModelType,
    pub model_id: String,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            session: get_model(),
            tokenizer: get_tokenizer(),
            config: get_model_config(),
            model_type: get_model_type(),
            model_id: get_model_id(),
        }
    }
}
