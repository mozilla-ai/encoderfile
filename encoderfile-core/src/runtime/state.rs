use std::{marker::PhantomData, sync::Arc};

use ort::session::Session;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

use crate::common::{ModelConfig, ModelType, model_type::ModelTypeSpec};

#[derive(Debug, Clone)]
pub struct AppState<T: ModelTypeSpec> {
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: Arc<ModelConfig>,
    // pub model_type: ModelType,
    pub model_id: String,
    pub transform_str: Option<String>,
    _marker: PhantomData<T>,
}

impl<T: ModelTypeSpec> AppState<T> {
    pub fn new(
        session: Arc<Mutex<Session>>,
        tokenizer: Arc<Tokenizer>,
        config: Arc<ModelConfig>,
        model_id: String,
        transform_str: Option<String>,
    ) -> AppState<T> {
        AppState {
            session,
            tokenizer,
            config,
            model_id,
            transform_str,
            _marker: PhantomData,
        }
    }

    pub fn transform_str(&self) -> Option<&str> {
        match &self.transform_str {
            Some(t) => Some(t.as_ref()),
            None => None,
        }
    }

    pub fn model_type() -> ModelType {
        T::enum_val()
    }
}
