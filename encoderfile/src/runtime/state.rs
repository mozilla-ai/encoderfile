use std::{marker::PhantomData, sync::Arc};

use ort::session::Session;
use parking_lot::Mutex;

use crate::{
    common::{Config, LuaLibs, ModelConfig, ModelType, model_type::ModelTypeSpec},
    runtime::TokenizerService,
    transforms::DEFAULT_LIBS,
};

pub type AppState<T> = Arc<EncoderfileState<T>>;

#[derive(Debug)]
pub struct EncoderfileState<T: ModelTypeSpec> {
    pub config: Config,
    pub session: Mutex<Session>,
    pub tokenizer: TokenizerService,
    pub model_config: ModelConfig,
    pub lua_libs: Vec<mlua::StdLib>,
    _marker: PhantomData<T>,
}

impl<T: ModelTypeSpec> EncoderfileState<T> {
    pub fn new(
        config: Config,
        session: Mutex<Session>,
        tokenizer: TokenizerService,
        model_config: ModelConfig,
    ) -> EncoderfileState<T> {
        let lua_libs = match config.lua_libs {
            Some(ref libs) => Vec::<mlua::StdLib>::from(libs),
            None => DEFAULT_LIBS.to_vec(),
        };
        EncoderfileState {
            config,
            session,
            tokenizer,
            model_config,
            lua_libs,
            _marker: PhantomData,
        }
    }

    pub fn transform_str(&self) -> Option<String> {
        self.config.transform.clone()
    }

    pub fn lua_options(&self) -> &Option<LuaLibs> {
        &self.config.lua_libs
    }

    pub fn model_type() -> ModelType {
        T::enum_val()
    }
}
