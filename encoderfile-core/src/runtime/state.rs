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
