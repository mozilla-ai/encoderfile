use std::path::PathBuf;

use crate::common::Config;

pub struct EncoderAssetPlan {
    runtime_config: Config,
    weights_path: PathBuf,
    tokenizer_path: PathBuf,
    model_config_path: PathBuf,
}

pub enum AssetKind {
    RuntimeConfig,
    ModelWeights,
    Tokenizer,
    ModelConfig,
}