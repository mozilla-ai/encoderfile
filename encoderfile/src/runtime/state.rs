use std::{marker::PhantomData, sync::Arc};

use ort::execution_providers as ep;
use ort::session::{Session, builder::SessionBuilder};
use parking_lot::Mutex;

use crate::{
    common::{Config, ModelConfig, ModelType, model_type::ModelTypeSpec},
    runtime::TokenizerService,
    transforms::DEFAULT_LIBS,
};

pub type AppState<T> = Arc<EncoderfileState<T>>;

// TODO allow options for the backend
pub fn assemble_ort_builder() -> Result<SessionBuilder, ort::Error> {
    SessionBuilder::new()?
        .with_execution_providers([
            // Prefer TensorRT over CUDA.
            ep::TensorRTExecutionProvider::default().build(),
            ep::CUDAExecutionProvider::default().build(),
            // Use DirectML on Windows if NVIDIA EPs are not available
            ep::DirectMLExecutionProvider::default().build(),
            // Or use ANE on Apple platforms
            ep::CoreMLExecutionProvider::default().build(),
        ])?
        .with_parallel_execution(true)?
        .with_inter_threads(4)
}

pub struct GlobalState {
    pub builder: Mutex<SessionBuilder>,
}

impl GlobalState {
    pub fn new() -> Result<Self, ort::Error> {
        Ok(Self {
            builder: Mutex::new(assemble_ort_builder()?),
        })
    }
}

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

    pub fn lua_libs(&self) -> &Vec<mlua::StdLib> {
        &self.lua_libs
    }

    pub fn model_type() -> ModelType {
        T::enum_val()
    }
}
