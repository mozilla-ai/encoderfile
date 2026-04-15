use crate::{
    common::{
        FromCliInput, ModelType,
        model_type::{self, ModelTypeSpec},
    },
    runtime::{EncoderfileLoader, EncoderfileState, ORTExecutionProvider},
    services::{Inference, Metadata},
    transport::{
        grpc::GrpcRouter,
        http::HttpRouter,
        mcp::McpRouter,
        server::{run_grpc, run_http, run_mcp},
    },
};
use anyhow::Result;
use clap_derive::{Args, Parser, Subcommand, ValueEnum};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::{
    fmt::Display,
    io::{Read, Seek, Write},
    sync::Arc,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub trait CliRoute: Inference {
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

impl<T: Inference> CliRoute for T {}

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
        #[command(flatten)]
        ep_args: ExecutionProviderArgs,
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
    pub async fn execute<'a, R: Read + Seek>(
        self,
        loader: &mut EncoderfileLoader<'a, R>,
    ) -> Result<()> {
        match loader.model_type() {
            ModelType::Embedding => {
                self.execute_from_loader::<R, model_type::Embedding>(loader)
                    .await
            }
            ModelType::SequenceClassification => {
                self.execute_from_loader::<R, model_type::SequenceClassification>(loader)
                    .await
            }
            ModelType::TokenClassification => {
                self.execute_from_loader::<R, model_type::TokenClassification>(loader)
                    .await
            }
            ModelType::SentenceEmbedding => {
                self.execute_from_loader::<R, model_type::SentenceEmbedding>(loader)
                    .await
            }
        }
    }
    pub async fn execute_from_loader<'a, R: Read + Seek, T: ModelTypeSpec>(
        self,
        loader: &mut EncoderfileLoader<'a, R>,
    ) -> Result<()>
    where
        Arc<EncoderfileState<T>>: Inference + GrpcRouter + HttpRouter + McpRouter + CliRoute,
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
                ep_args,
            } => {
                let session = loader.session()?.into();
                let model_config = loader.model_config()?;
                let tokenizer = loader.tokenizer()?;
                let config = loader.encoderfile_config()?;

                let state = Arc::new(EncoderfileState::<T>::new(
                    config,
                    session,
                    tokenizer,
                    model_config,
                ));

                let banner = crate::get_banner(state.model_id().as_str());

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
                let session = loader.session()?.into();
                let model_config = loader.model_config()?;
                let tokenizer = loader.tokenizer()?;
                let config = loader.encoderfile_config()?;

                let state = Arc::new(EncoderfileState::<T>::new(
                    config,
                    session,
                    tokenizer,
                    model_config,
                ));

                setup_tracing(None)?;

                state.cli_route(inputs, format, out_dir)?
            }
            Commands::Mcp {
                hostname,
                port,
                cert_file,
                key_file,
            } => {
                let session = loader.session()?.into();
                let model_config = loader.model_config()?;
                let tokenizer = loader.tokenizer()?;
                let config = loader.encoderfile_config()?;

                let state = Arc::new(EncoderfileState::<T>::new(
                    config,
                    session,
                    tokenizer,
                    model_config,
                ));

                let banner = crate::get_banner(state.model_id().as_str());
                let mcp_process = tokio::spawn(run_mcp(hostname, port, cert_file, key_file, state));
                println!("{}", banner);
                let _ = tokio::join!(mcp_process);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Args)]
pub struct ExecutionProviderArgs {
    #[arg(long, default_value_t = ExecutionProvider::Cpu)]
    execution_provider: ExecutionProvider,
    #[arg(
        long,
        default_value_t = false,
        help = "Use arena allocator. Use only if execution provider is CPU."
    )]
    with_arena_allocator: bool,
    #[arg(
        long,
        help = "Device. Use only if execution provider is CUDA/TensorRT."
    )]
    device_id: Option<i32>,
    #[arg(
        long,
        default_value_t = CoreMLComputeUnits::All,
        help = "Compute units. Use only if execution provider is CoreML."
    )]
    compute_units: CoreMLComputeUnits,
}

impl ExecutionProviderArgs {
    pub fn to_provider(&self) -> ORTExecutionProvider {
        match self.execution_provider {
            ExecutionProvider::Cpu => ORTExecutionProvider::Cpu {
                arena_allocator: self.with_arena_allocator,
            },
            ExecutionProvider::Cuda => ORTExecutionProvider::Cuda {
                device_id: self.device_id,
            },
            ExecutionProvider::Tensorrt => ORTExecutionProvider::TensorRT {
                device_id: self.device_id,
            },
            ExecutionProvider::Coreml => ORTExecutionProvider::CoreML {
                compute_units: Some(match self.compute_units {
                    CoreMLComputeUnits::All => {
                        ort::execution_providers::coreml::CoreMLComputeUnits::All
                    }
                    CoreMLComputeUnits::CpuAndNeuralEngine => {
                        ort::execution_providers::coreml::CoreMLComputeUnits::CPUAndNeuralEngine
                    }
                    CoreMLComputeUnits::CpuAndGpu => {
                        ort::execution_providers::coreml::CoreMLComputeUnits::CPUAndGPU
                    }
                    CoreMLComputeUnits::CpuOnly => {
                        ort::execution_providers::coreml::CoreMLComputeUnits::CPUOnly
                    }
                }),
            },
        }
    }
}

#[derive(Clone, ValueEnum)]
pub enum CoreMLComputeUnits {
    All,
    CpuAndNeuralEngine,
    CpuAndGpu,
    CpuOnly,
}

impl Display for CoreMLComputeUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreMLComputeUnits::All => write!(f, "all"),
            CoreMLComputeUnits::CpuAndNeuralEngine => write!(f, "cpu-and-neural-engine"),
            CoreMLComputeUnits::CpuAndGpu => write!(f, "cpu-and-gpu"),
            CoreMLComputeUnits::CpuOnly => write!(f, "cpu-only"),
        }
    }
}

#[derive(Clone, ValueEnum)]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
    Tensorrt,
    Coreml,
}

impl Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionProvider::Cpu => write!(f, "cpu"),
            ExecutionProvider::Cuda => write!(f, "cuda"),
            ExecutionProvider::Tensorrt => write!(f, "tensorrt"),
            ExecutionProvider::Coreml => write!(f, "coreml"),
        }
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
