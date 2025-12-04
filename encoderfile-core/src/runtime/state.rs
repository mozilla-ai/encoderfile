use std::sync::Arc;

use ort::session::Session;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use crate::common::{ModelConfig, ModelType};

#[derive(Debug, Clone)]
pub struct AppState {
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: Arc<ModelConfig>,
    pub model_type: ModelType,
    pub model_id: String,
    transform_str: Option<String>,
}

impl AppState {
    pub fn transform_str(&self) -> Option<&str> {
        match &self.transform_str {
            Some(t) => Some(t.as_ref()),
            None => None,
        }
    }
}
