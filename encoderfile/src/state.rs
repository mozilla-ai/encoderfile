use std::sync::Arc;

use ort::session::Session;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use crate::{config::{ModelConfig, ModelType, get_model_config, get_model_type}, inference::{model::get_model, tokenizer::get_tokenizer}};

#[derive(Debug, Clone)]
pub struct AppState {
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: &'static Tokenizer,
    pub config: &'static ModelConfig,
    pub model_type: &'static ModelType,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            session: get_model(),
            tokenizer: get_tokenizer(),
            config: get_model_config(),
            model_type: get_model_type(),
        }
    }
}
