use crate::{
    common::{
        EmbeddingRequest, ModelType, SequenceClassificationRequest, TokenClassificationRequest,
    },
    server::{run_grpc, run_http, run_mcp},
    config::get_model_type,
    services::{embedding, sequence_classification, token_classification},
};
use anyhow::Result;
use clap_derive::{Parser, Subcommand, ValueEnum};
use std::io::Write;

macro_rules! generate_cli_route {
    ($req:ident, $fn:path, $format:ident, $out_dir:expr) => {{
        let state = $crate::state::AppState::default();
        let result = $fn($req, &state)?;

        let serialized = match $format {
            Format::Json => serde_json::to_string_pretty(&result)?,
        };

        match $out_dir {
            Some(o) => {
                let mut file = std::fs::File::create(o)?;
                file.write_all(serialized.as_bytes())?;
            }
            None => println!("{}", serialized),
        }
    }};
}

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Serve {
        #[arg(long, default_value = "[::]")]
        grpc_hostname: String,
        #[arg(long, default_value = "50051")]
        grpc_port: String,
        #[arg(long, default_value = "0.0.0.0")]
        http_hostname: String,
        #[arg(long, default_value = "8080")]
        http_port: String,
        #[arg(long, default_value_t = false)]
        disable_grpc: bool,
        #[arg(long, default_value_t = false)]
        disable_http: bool,
        #[arg(long, default_value = "0.0.0.0")]
        mcp_hostname: String,
        #[arg(long, default_value = "9100")]
        mcp_port: String,
        #[arg(long, default_value_t = false)]
        disable_mcp: bool,
    },
    Infer {
        #[arg(required = true)]
        inputs: Vec<String>,
        #[arg(long, default_value_t = true)]
        normalize: bool,
        #[arg(short, long, default_value_t = Format::Json)]
        format: Format,
        #[arg(short)]
        out_dir: Option<String>,
    },
}

impl Commands {
    pub async fn execute(self) -> Result<()> {
        match self {
            Commands::Serve {
                grpc_hostname,
                grpc_port,
                http_hostname,
                http_port,
                disable_grpc,
                disable_http,
                mcp_hostname,
                mcp_port,
                disable_mcp,
            } => {
                if disable_grpc && disable_http {
                    return Err(crate::error::ApiError::ConfigError(
                        "Cannot disable both gRPC and HTTP",
                    ))?;
                }

                let grpc_process = match disable_grpc {
                    true => tokio::spawn(async { Ok(()) }),
                    false => tokio::spawn(run_grpc(grpc_hostname, grpc_port)),
                };

                let http_process = match disable_http {
                    true => tokio::spawn(async { Ok(()) }),
                    false => tokio::spawn(run_http(http_hostname, http_port)),
                };

                let mcp_process = match disable_mcp {
                    true => tokio::spawn(async { Ok(()) }),
                    false => tokio::spawn(run_mcp(mcp_hostname, mcp_port)),
                };

                println!("{}", crate::get_banner());

                let _ = tokio::join!(grpc_process, http_process);
            }
            Commands::Infer {
                inputs,
                normalize,
                format,
                out_dir,
            } => {
                let metadata = None;

                match get_model_type() {
                    ModelType::Embedding => {
                        let request = EmbeddingRequest {
                            inputs,
                            normalize,
                            metadata,
                        };

                        generate_cli_route!(request, embedding, format, out_dir)
                    }
                    ModelType::SequenceClassification => {
                        let request = SequenceClassificationRequest { inputs, metadata };

                        generate_cli_route!(request, sequence_classification, format, out_dir)
                    }
                    ModelType::TokenClassification => {
                        let request = TokenClassificationRequest { inputs, metadata };

                        generate_cli_route!(request, token_classification, format, out_dir)
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, ValueEnum)]
pub enum Format {
    Json,
}

impl ToString for Format {
    fn to_string(&self) -> String {
        match self {
            Format::Json => "json",
        }
        .to_string()
    }
}
