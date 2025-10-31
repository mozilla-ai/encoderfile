use crate::{
    config::{ModelType, get_model_type},
    server::{run_grpc, run_http},
    services::{
        SequenceClassificationRequest, TokenClassificationRequest, embedding,
        sequence_classification, token_classification,
    },
};
use anyhow::Result;
use clap_derive::{Parser, Subcommand};
use serde_json::json;

macro_rules! generate_cli_route {
    ($req:ident, $fn:path) => {
        match $fn($req) {
            Ok(r) => println!("{}", json!(r).to_string()),
            Err(e) => println!("{}", json!(e).to_string()),
        }
    };
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
    },
    Infer {
        #[arg(required = true)]
        inputs: Vec<String>,
        #[arg(long, default_value_t = true)]
        normalize: bool,
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

                println!("{}", crate::get_banner());

                let _ = tokio::join!(grpc_process, http_process);
            }
            Commands::Infer { inputs, normalize } => {
                let metadata = None;

                match get_model_type() {
                    ModelType::Embedding => {
                        let request = crate::services::EmbeddingRequest {
                            inputs,
                            normalize,
                            metadata,
                        };

                        generate_cli_route!(request, embedding)
                    }
                    ModelType::SequenceClassification => {
                        let request = SequenceClassificationRequest { inputs, metadata };

                        generate_cli_route!(request, sequence_classification)
                    }
                    ModelType::TokenClassification => {
                        let request = TokenClassificationRequest { inputs, metadata };

                        generate_cli_route!(request, token_classification)
                    }
                }
            }
        }
        Ok(())
    }
}
