use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EncoderfileBuildError {
    #[error("Failed to resolve build target: {0}")]
    Target(#[from] super::base_binary::TargetError),
    #[error("Failed to resolve base binary: {0}")]
    BaseBinary(#[from] super::base_binary::BaseBinaryError),
    #[error("failed to serialize model config: {0}")]
    ConfigSerialization(#[source] serde_json::Error),
    #[error("failed to write encoderfile to {path}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    // TODO: Update with AssetValidationError
    #[error("Asset validation error: {0}")]
    AssetValidation(#[from] anyhow::Error),
}
