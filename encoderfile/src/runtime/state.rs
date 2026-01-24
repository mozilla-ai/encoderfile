use std::{marker::PhantomData, sync::Arc};

use ort::session::Session;
use parking_lot::Mutex;

use crate::{
    common::{Config, ModelConfig, ModelType, model_type::ModelTypeSpec},
    runtime::TokenizerService,
};

pub type AppState<T> = Arc<EncoderfileState<T>>;

#[derive(Debug)]
pub struct EncoderfileState<T: ModelTypeSpec> {
    pub config: Config,
    pub session: Mutex<Session>,
    pub tokenizer: TokenizerService,
    pub model_config: ModelConfig,
    _marker: PhantomData<T>,
}

impl<T: ModelTypeSpec> EncoderfileState<T> {
    pub fn new(
        config: Config,
        session: Mutex<Session>,
        tokenizer: TokenizerService,
        model_config: ModelConfig,
    ) -> EncoderfileState<T> {
        EncoderfileState {
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
