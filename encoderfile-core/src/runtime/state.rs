use std::sync::Arc;

use ort::session::Session;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use crate::common::{ModelConfig, ModelTypeEnum};

#[derive(Debug, Clone)]
pub struct AppState {
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: Arc<ModelConfig>,
    pub model_type: ModelTypeEnum,
    pub model_id: String,
    pub transform_str: Option<String>,
}

impl AppState {
    pub fn transform_str(&self) -> Option<&str> {
        match &self.transform_str {
            Some(t) => Some(t.as_ref()),
            None => None,
        }
    }
}
