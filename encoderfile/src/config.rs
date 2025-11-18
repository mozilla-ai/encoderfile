use anyhow::{Result, bail};
use std::{io::Read, path::PathBuf};

use figment::{
    Figment,
    providers::{Format, Yaml},
};
use serde::{Deserialize, Serialize};
use tera::Context;
use sha2::{Sha256, Digest};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub encoderfile: EncoderfileConfig,
}

impl Config {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let config = Figment::new().merge(Yaml::file(path)).extract()?;

        Ok(config)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EncoderfileConfig {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    pub path: ModelPath,
    pub model_type: ModelType,
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,
    pub transform: Option<Transform>,
    #[serde(default = "default_build")]
    pub build: bool
}

impl EncoderfileConfig {
    pub fn to_tera_ctx(&self) -> Result<Context> {
        let mut ctx = Context::new();

        let transform = match &self.transform {
            None => None,
            Some(s) => Some(s.transform()?),
        };

        ctx.insert("version", self.version.as_str());
        ctx.insert("model_name", self.name.as_str());
        ctx.insert("model_type", &self.model_type);
        ctx.insert("model_weights_path", &self.path.model_weights_path()?);
        ctx.insert("tokenizer_path", &self.path.tokenizer_path()?);
        ctx.insert("model_config_path", &self.path.model_config_path()?);
        ctx.insert("transform", &transform);

        Ok(ctx)
    }

    pub fn get_write_dir(&self) -> PathBuf {
        let id = uuid::Uuid::new_v4().to_string();

        let filename_hash = Sha256::digest(self.name.as_bytes());

        self.cache_dir.join(format!("encoderfile-{:x}-{}", filename_hash, id))
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Transform {
    Path { path: PathBuf },
    Inline(String),
}

impl Transform {
    pub fn transform(&self) -> Result<String> {
        match self {
            Self::Path { path } => {
                if !path.exists() {
                    bail!("No such file: {:?}", &path);
                }

                let mut code = String::new();

                std::fs::File::open(path)?.read_to_string(&mut code)?;

                Ok(code)
            }
            Self::Inline(s) => Ok(s.clone()),
        }
        .map(|i| i.trim().to_string())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelPath {
    Directory(PathBuf),
    Paths {
        model_config_path: PathBuf,
        model_weights_path: PathBuf,
        tokenizer_path: PathBuf,
    },
}

macro_rules! asset_path {
    ($var:ident, $default:expr, $err:expr) => {
        pub fn $var(&self) -> Result<PathBuf> {
            let path = match self {
                Self::Paths { $var, .. } => $var.clone(),
                Self::Directory(dir) => {
                    if !dir.is_dir() {
                        bail!("No such directory: {:?}", dir);
                    }
                    dir.join($default)
                }
            };

            if !path.try_exists()? {
                bail!("Could not locate {} at path: {:?}", $err, path);
            }

            Ok(path.canonicalize()?)
        }
    };
}

impl ModelPath {
    asset_path!(model_config_path, "config.json", "model config");
    asset_path!(tokenizer_path, "tokenizer.json", "tokenizer");
    asset_path!(model_weights_path, "model.onnx", "model weights");
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Embedding,
    SequenceClassification,
    TokenClassification,
    SentenceEmbedding,
}

fn default_output_dir() -> PathBuf {
    std::env::current_dir()
        .expect("Can't even find the current dir? Tragic. (no seriously please open an issue)")
}

fn default_cache_dir() -> PathBuf {
    directories::ProjectDirs::from("com", "mozilla-ai", "encoderfile")
        .expect("Cannot locate")
        .cache_dir()
        .to_path_buf()
}

fn default_version() -> String {
    "0.1.0".to_string()
}

fn default_build() -> bool {
    true
}
