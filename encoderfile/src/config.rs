use anyhow::{Result, bail};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct EncoderfileConfig {
    name: String,
    path: ModelPath,
    model_type: ModelType,
    #[serde(default = "default_output_dir")]
    output_dir: PathBuf,
    #[serde(default = "default_cache_dir")]
    cache_dir: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
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

            Ok(path)
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
