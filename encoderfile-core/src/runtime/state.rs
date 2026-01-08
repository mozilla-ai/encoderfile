use anyhow::{Result, bail};
use prost::Message;
use std::{
    io::{Read, Seek},
    marker::PhantomData,
    sync::Arc,
};

use ort::session::Session;
use parking_lot::{Mutex, RawMutex, lock_api::MutexGuard};

use crate::{
    common::{
        Config, ModelConfig, ModelType, TokenizerConfig,
        model_type::{
            Embedding, ModelTypeSpec, SentenceEmbedding, SequenceClassification,
            TokenClassification,
        },
    },
    format::{
        assets::{AssetKind, AssetPolicySpec},
        container::Encoderfile,
    },
    generated::manifest::TransformType,
    runtime::TokenizerService,
};

#[derive(Debug)]
pub struct AppStateLoader<'a, R: Read + Seek> {
    encoderfile: &'a Encoderfile,
    reader: &'a mut R,
}

impl<'a, R> AppStateLoader<'a, R>
where
    R: Read + Seek,
{
    pub fn load<T: AssetPolicySpec>(self) -> Result<Box<dyn InferenceState>> {
        let transform = match self
            .encoderfile
            .open_optional(self.reader, AssetKind::Transform)
        {
            Some(mut r) => {
                let mut buf = vec![0u8; r.len() as usize];
                r.read_exact(&mut buf)?;

                let transform_proto = crate::generated::manifest::Transform::decode(&*buf)?;

                // NOTE: update if we ever support other transform types besides Lua (unlikely)
                match transform_proto.transform_type() {
                    TransformType::Lua => (),
                    TransformType::UndeclaredTransform => {
                        bail!("Unspecified transform type. This should not happen.")
                    }
                };

                Some(transform_proto.transform)
            }
            None => None,
        };

        let config = Arc::new(Config {
            name: self.encoderfile.name().to_string(),
            version: self.encoderfile.version().to_string(),
            model_type: self.encoderfile.model_type(),
            transform,
            // TODO: remove as we don't use this anymore
            tokenizer: TokenizerConfig::default(),
        });

        let session = match self
            .encoderfile
            .open_required(self.reader, AssetKind::ModelWeights)
        {
            Ok(mut r) => {
                let mut buf = vec![0u8; r.len() as usize];
                r.read_exact(&mut buf)?;

                let session =
                    ort::session::Session::builder()?.commit_from_memory(buf.as_slice())?;
                Arc::new(Mutex::new(session))
            }
            Err(e) => bail!("Error loading model weights: {e:?}"),
        };

        let tokenizer = match self
            .encoderfile
            .open_required(self.reader, AssetKind::Tokenizer)
        {
            Ok(mut r) => {
                let mut buf = vec![0u8; r.len() as usize];
                r.read_exact(&mut buf)?;

                let tokenizer = serde_json::from_slice(buf.as_slice())?;

                Arc::new(tokenizer)
            }
            Err(e) => bail!("Error loading tokenizer: {e:?}"),
        };

        let model_config = match self
            .encoderfile
            .open_required(self.reader, AssetKind::ModelConfig)
        {
            Ok(mut r) => {
                let mut buf = vec![0u8; r.len() as usize];
                r.read_exact(&mut buf)?;

                let model_config = serde_json::from_slice(buf.as_slice())?;

                Arc::new(model_config)
            }
            Err(e) => bail!("Error loading model config: {e:?}"),
        };

        let state: Box<dyn InferenceState> = match self.encoderfile.model_type() {
            ModelType::Embedding => Box::new(AppState::<Embedding>::new(
                config,
                session,
                tokenizer,
                model_config,
            )),
            ModelType::SequenceClassification => Box::new(AppState::<SequenceClassification>::new(
                config,
                session,
                tokenizer,
                model_config,
            )),
            ModelType::TokenClassification => Box::new(AppState::<TokenClassification>::new(
                config,
                session,
                tokenizer,
                model_config,
            )),
            ModelType::SentenceEmbedding => Box::new(AppState::<SentenceEmbedding>::new(
                config,
                session,
                tokenizer,
                model_config,
            )),
        };

        Ok(state)
    }
}

pub trait InferenceState {
    fn config(&self) -> &Arc<Config>;
    fn session(&self) -> MutexGuard<'_, RawMutex, Session>;
    fn tokenizer(&self) -> &Arc<TokenizerService>;
    fn model_config(&self) -> &Arc<ModelConfig>;
}

impl<T: ModelTypeSpec> InferenceState for AppState<T> {
    fn config(&self) -> &Arc<Config> {
        &self.config
    }
    fn session(&self) -> MutexGuard<'_, RawMutex, Session> {
        self.session.lock()
    }
    fn tokenizer(&self) -> &Arc<TokenizerService> {
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
