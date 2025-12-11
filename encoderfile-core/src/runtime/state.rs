use std::{marker::PhantomData, sync::Arc};

use ort::session::Session;
use parking_lot::{Mutex, RawMutex, lock_api::MutexGuard};
use tokenizers::Tokenizer;

use crate::common::{Config, ModelConfig, ModelType, model_type::ModelTypeSpec};

pub trait InferenceState {
    fn config(&self) -> &Arc<Config>;
    fn session(&self) -> MutexGuard<'_, RawMutex, Session>;
    fn tokenizer(&self) -> &Arc<Tokenizer>;
    fn model_config(&self) -> &Arc<ModelConfig>;
}

impl<T: ModelTypeSpec> InferenceState for AppState<T> {
    fn config(&self) -> &Arc<Config> {
        &self.config
    }
    fn session(&self) -> MutexGuard<'_, RawMutex, Session> {
        self.session.lock()
    }
    fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }
    fn model_config(&self) -> &Arc<ModelConfig> {
        &self.model_config
    }
}

#[derive(Debug, Clone)]
pub struct AppState<T: ModelTypeSpec> {
    pub config: Arc<Config>,
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_config: Arc<ModelConfig>,
    _marker: PhantomData<T>,
}

impl<T: ModelTypeSpec> AppState<T> {
    pub fn new(
        config: Arc<Config>,
        session: Arc<Mutex<Session>>,
        tokenizer: Arc<Tokenizer>,
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
