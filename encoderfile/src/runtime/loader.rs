use anyhow::{Result, bail};
use prost::Message;
use std::io::{Read, Seek};

use ort::session::Session;

use crate::{
    common::{Config, LuaLibs, ModelConfig, ModelType},
    format::{assets::AssetKind, codec::EncoderfileCodec, container::Encoderfile},
    generated::manifest::{self, TransformType},
    runtime::TokenizerService,
};

pub struct EncoderfileLoader<'a, R: Read + Seek> {
    encoderfile: Encoderfile,
    reader: &'a mut R,
}

impl<'a, R: Read + Seek> EncoderfileLoader<'a, R> {
    pub fn new(encoderfile: Encoderfile, reader: &'a mut R) -> Self {
        Self {
            encoderfile,
            reader,
        }
    }

    pub fn model_type(&self) -> ModelType {
        self.encoderfile.model_type()
    }

    pub fn session(&mut self) -> Result<Session> {
        let session = match self
            .encoderfile
            .open_required(self.reader, AssetKind::ModelWeights)
        {
            Ok(mut r) => {
                let mut buf = vec![0u8; r.len() as usize];
                r.read_exact(&mut buf)?;

                ort::session::Session::builder()?.commit_from_memory(buf.as_slice())?
            }
            Err(e) => bail!("Error loading model weights: {e:?}"),
        };

        Ok(session)
    }

    pub fn tokenizer(&mut self) -> Result<TokenizerService> {
        match self
            .encoderfile
            .open_required(self.reader, AssetKind::Tokenizer)
        {
            Ok(mut r) => {
                let mut buf = vec![0u8; r.len() as usize];
                r.read_exact(&mut buf)?;

                Ok(serde_json::from_slice(buf.as_slice())?)
            }
            Err(e) => bail!("Error loading tokenizer: {e:?}"),
        }
    }

    pub fn transform(&mut self) -> Result<Option<manifest::Transform>> {
        let transform_proto = match self
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
                    TransformType::Unspecified => {
                        bail!("Unspecified transform type. This should not happen.")
                    }
                };

                Some(transform_proto)
            }
            None => None,
        };

        Ok(transform_proto)
    }

    pub fn encoderfile_config(&mut self) -> Result<Config> {
        let transform = self.transform()?;
        let protolibs = transform
            .as_ref()
            .and_then(|t| t.lua_libs.clone())
            .map(|l| l.libs);
        let configlibs = protolibs.map(LuaLibs::try_from).transpose()?;
        let config = Config {
            name: self.encoderfile.name().to_string(),
            version: self.encoderfile.version().to_string(),
            model_type: self.encoderfile.model_type(),
            transform: transform.map(|t| t.transform),
            lua_libs: configlibs,
        };
        Ok(config)
    }

    pub fn model_config(&mut self) -> Result<ModelConfig> {
        match self
            .encoderfile
            .open_required(self.reader, AssetKind::ModelConfig)
        {
            Ok(mut r) => {
                let mut buf = vec![0u8; r.len() as usize];
                r.read_exact(&mut buf)?;

                Ok(serde_json::from_slice(buf.as_slice())?)
            }
            Err(e) => bail!("Error loading model config: {e:?}"),
        }
    }
}

pub fn load_assets<'a, R: Read + Seek>(file: &'a mut R) -> Result<EncoderfileLoader<'a, R>> {
    let encoderfile = EncoderfileCodec::read(file)?;
    let loader = EncoderfileLoader::new(encoderfile, file);

    Ok(loader)
}
