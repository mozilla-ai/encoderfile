use crate::{
    common::{FromCliInput, model_type::ModelTypeSpec},
    runtime::AppState,
    services::Inference,
    transport::{
        grpc::GrpcRouter,
        http::HttpRouter,
        mcp::McpRouter,
        server::{run_grpc, run_http, run_mcp},
    },
};
use anyhow::Result;
use clap_derive::{Parser, Subcommand, ValueEnum};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::{fmt::Display, io::Write};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

trait CliRoute: Inference {
    fn cli_route(
        &self,
        inputs: Vec<String>,
        format: Format,
        out_dir: Option<String>,
    ) -> Result<()> {
        let input = Self::Input::from_cli_input(inputs);

        let result = self.inference(input)?;

        let serialized = match format {
            Format::Json => serde_json::to_string_pretty(&result)?,
        };

        match out_dir {
            Some(o) => {
                let mut file = std::fs::File::create(o)?;
                file.write_all(serialized.as_bytes())?;
            }
            None => println!("{}", serialized),
        };

        Ok(())
    }
}

impl<T: ModelTypeSpec> CliRoute for AppState<T> where AppState<T>: Inference {}

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
        #[arg(long, default_value_t = false)]
        enable_otel: bool,
        #[arg(long, default_value = "http://localhost:4317")]
        otel_exporter_url: String,
        #[arg(long)]
        cert_file: Option<String>,
        #[arg(long)]
        key_file: Option<String>,
    },
    Infer {
        #[arg(required = true)]
        inputs: Vec<String>,
        #[arg(short, long, default_value_t = Format::Json)]
        format: Format,
        #[arg(short)]
        out_dir: Option<String>,
    },
    Mcp {
        #[arg(long, default_value = "0.0.0.0")]
        hostname: String,
        #[arg(long, default_value = "9100")]
        port: String,
        #[arg(long)]
        cert_file: Option<String>,
        #[arg(long)]
        key_file: Option<String>,
    },
}

impl Commands {
    pub async fn execute<T: ModelTypeSpec + GrpcRouter + McpRouter + HttpRouter>(
        self,
        state: AppState<T>,
    ) -> Result<()>
    where
        AppState<T>: Inference,
    {
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
                cert_file,
                key_file,
            } => {
                let banner = crate::get_banner(state.config.name.as_str());

                if disable_grpc && disable_http {
                    return Err(crate::error::ApiError::ConfigError(
                        "Cannot disable both gRPC and HTTP",
                    ))?;
                }

                match enable_otel {
                    true => setup_tracing(Some(otel_exporter_url.as_str())),
                    false => setup_tracing(None),
                }?;

                let grpc_process = match disable_grpc {
                    true => tokio::spawn(async { Ok(()) }),
                    false => tokio::spawn(run_grpc(
                        grpc_hostname,
                        grpc_port,
                        cert_file.clone(),
                        key_file.clone(),
                        state.clone(),
                    )),
                };

                let http_process = match disable_http {
                    true => tokio::spawn(async { Ok(()) }),
                    false => tokio::spawn(run_http(
                        http_hostname,
                        http_port,
                        cert_file.clone(),
                        key_file.clone(),
                        state.clone(),
                    )),
                };

                println!("{}", banner);

                let _ = tokio::join!(grpc_process, http_process);
            }
            Commands::Infer {
                inputs,
                format,
                out_dir,
            } => {
                setup_tracing(None)?;

                state.cli_route(inputs, format, out_dir)?
            }
            Commands::Mcp {
                hostname,
                port,
                cert_file,
                key_file,
            } => {
                let banner = crate::get_banner(state.config.name.as_str());
                let mcp_process = tokio::spawn(run_mcp(hostname, port, cert_file, key_file, state));
                println!("{}", banner);
                let _ = tokio::join!(mcp_process);
            }
        }
        Ok(())
    }
}

#[derive(Clone, ValueEnum)]
pub enum Format {
    Json,
}

impl Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Format::Json => write!(f, "json"),
        }
    }
}

fn setup_tracing(otlp_exporter_url: Option<&str>) -> anyhow::Result<()> {
    if let Some(otlp_exporter_url) = otlp_exporter_url {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_protocol(Protocol::Grpc)
            .with_endpoint(otlp_exporter_url)
            .build()?;

        let resource = opentelemetry_sdk::Resource::builder()
            .with_attributes(vec![opentelemetry::KeyValue::new(
                opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                "encoderfile",
            )])
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
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn")), // default to "info" level
            )
            .with_target(false) // hide module path
            .compact() // short, pretty output
            .init();
    }

    Ok(())
}
