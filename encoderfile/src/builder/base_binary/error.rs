use reqwest::Url;
use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum BaseBinaryError {
    #[error("cannot remove an explicitly provided base binary path")]
    CannotRemoveExplicitPath,

    #[error("downloads disabled but base binary is not cached")]
    DownloadDisabled,

    #[error("base binary missing at {path}")]
    BinaryMissing {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("base binary at {0} is not a regular file")]
    NotAFile(PathBuf),

    #[error("base binary at {0} is not executable")]
    NotExecutable(PathBuf),

    #[error("download failed with status {status} for {url}")]
    DownloadStatus {
        status: reqwest::StatusCode,
        url: Url,
    },

    #[error("failed to download {url}")]
    DownloadRequest {
        url: Url,
        #[source]
        source: reqwest::Error,
    },

    #[error("archive did not contain `{0}`")]
    ArchiveMissingRuntime(&'static str),

    #[error("failed to extract archive")]
    ArchiveExtract(#[source] std::io::Error),

    #[error("invalid {env_var} `{raw}`")]
    InvalidBaseUrlOverride {
        env_var: &'static str,
        raw: String,
        #[source]
        source: url::ParseError,
    },

    #[error("failed to construct download url")]
    UrlConstruction(#[source] url::ParseError),

    #[error("filesystem error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum TargetError {
    #[error("invalid or unsupported target triple `{0}`")]
    InvalidTriple(String),

    #[error("unsupported architecture `{0}`")]
    UnsupportedArch(String),

    #[error("unsupported operating system `{0}`")]
    UnsupportedOs(String),

    #[error("unsupported ABI `{abi}` for {os}")]
    UnsupportedAbi { abi: String, os: &'static str },

    #[error("architecture `{arch}` is not supported on {os}")]
    UnsupportedArchForOs { arch: String, os: &'static str },
}
