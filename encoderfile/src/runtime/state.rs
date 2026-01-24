use std::{marker::PhantomData, sync::Arc};

use ort::session::Session;
use parking_lot::Mutex;

use crate::{
    common::{Config, ModelConfig, ModelType, model_type::ModelTypeSpec},
    runtime::TokenizerService,
};

#[derive(Debug, Clone)]
pub struct AppState<T: ModelTypeSpec> {
    pub config: Arc<Config>,
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<TokenizerService>,
    pub model_config: Arc<ModelConfig>,
    _marker: PhantomData<T>,
}

impl<T: ModelTypeSpec> AppState<T> {
    pub fn new(
        config: Arc<Config>,
        session: Arc<Mutex<Session>>,
        tokenizer: Arc<TokenizerService>,
        model_config: Arc<ModelConfig>,
    ) -> AppState<T> {
        AppState {
            config,
            session,
            tokenizer,
            model_config,
            _marker: PhantomData,
        }
    }

    pub fn transform_str(&self) -> Option<String> {
        self.config.transform.clone()
    }

    pub fn model_type() -> ModelType {
        T::enum_val()
    }
}
