use crate::{
    common::{
        EmbeddingRequest, ModelType, SequenceClassificationRequest, TokenClassificationRequest,
    },
    model::config::get_model_type,
    server::{run_grpc, run_http},
    services::{embedding, sequence_classification, token_classification},
};
use anyhow::Result;
use clap_derive::{Parser, Subcommand, ValueEnum};
use std::io::Write;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry::trace::TracerProvider as _;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use opentelemetry_otlp::{WithExportConfig, Protocol};

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
        #[arg(long, default_value_t = true)]
        enable_otel: bool,
        #[arg(long, default_value = "http://localhost:4317")]
        otel_exporter_url: String,
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
                enable_otel,
                otel_exporter_url,
            } => {
                if disable_grpc && disable_http {
                    return Err(crate::error::ApiError::ConfigError(
                        "Cannot disable both gRPC and HTTP",
                    ))?;
                }

                match enable_otel {
                    true => setup_tracing(Some(otel_exporter_url.as_str())),
                    false => setup_tracing(None)
                }?;

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
            Commands::Infer {
                inputs,
                normalize,
                format,
                out_dir,
            } => {
                setup_tracing(None)?;

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

fn setup_tracing(otlp_exporter_url: Option<&str>) -> anyhow::Result<()> {
    if let Some(otlp_exporter_url) = otlp_exporter_url {

        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(otlp_exporter_url)
            .build()?;

        let resource = opentelemetry_sdk::Resource::builder()
            .with_attributes(vec![
                opentelemetry::KeyValue::new(opentelemetry_semantic_conventions::resource::SERVICE_NAME, "encoderfile"),
            ])
            .build();

        let provider = SdkTracerProvider::builder()
            .with_resource(resource)
            .with_batch_exporter(exporter)
            .build();

        let tracer = provider.tracer("encoderfile");

        // Create a tracing layer with the configured tracer
        let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

        let fmt_layer = tracing_subscriber::fmt::layer();
        let filter_layer = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn"));

        tracing_subscriber::registry()
            .with(filter_layer)
            .with(fmt_layer)
            .with(telemetry)
            .init();
    } else {
        tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,tower_http=debug,ort=warn")), // default to "info" level
        )
        .with_target(false) // hide module path
        .compact() // short, pretty output
        .init();
    }

    Ok(())
}
