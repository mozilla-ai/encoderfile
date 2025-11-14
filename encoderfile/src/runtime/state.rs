use std::sync::Arc;

use ort::session::Session;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use crate::{
    assets::get_model_id,
    common::ModelType,
    runtime::{
        config::{ModelConfig, get_model_config, get_model_type},
        model::get_model,
        tokenizer::get_tokenizer,
        transform::get_transform,
    },
    transforms::Transform,
};

#[derive(Debug, Clone)]
pub struct AppState {
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: Arc<ModelConfig>,
    pub model_type: ModelType,
    pub model_id: String,
    pub transform_factory: fn() -> Transform,
}

impl AppState {
    pub fn transform(&self) -> Transform {
        (self.transform_factory)()
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            session: get_model(),
            tokenizer: get_tokenizer(),
            config: get_model_config(),
            model_type: get_model_type(),
            model_id: get_model_id(),
            transform_factory: get_transform,
        }
    }
}
