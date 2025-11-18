use std::sync::Arc;

use ort::session::Session;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use crate::{common::ModelType, runtime::config::ModelConfig, transforms::Transform};

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
            session: crate::assets::get_model(),
            tokenizer: crate::assets::get_tokenizer(),
            config: crate::assets::get_model_config(),
            model_type: crate::assets::get_model_type(),
            model_id: crate::assets::get_model_id(),
            transform_factory: crate::assets::get_transform,
        }
    }
}
