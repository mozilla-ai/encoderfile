use thiserror::Error;

#[derive(Debug, Error)]
pub enum EncoderfileBuildError {
    #[error("Failed to resolve build target: {0}")]
    Target(#[from] super::base_binary::TargetError),
    #[error("Failed to resolve base binary: {0}")]
    BaseBinary(#[from] super::base_binary::BaseBinaryError),
}
