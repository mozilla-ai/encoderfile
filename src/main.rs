use anyhow::Result;
use clap::Parser;
use encoderfile::{cli::Commands, config::get_model_type, error::ApiError};
use tracing_subscriber::EnvFilter;

async fn run_grpc(hostname: String, port: String) -> Result<()> {
    let addr = format!("{}:{}", &hostname, &port);

    let router = encoderfile::grpc::router()
        .layer(tower_http::trace::TraceLayer::new_for_grpc())
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    tracing::info!("Running {:?} gRPC server on {}", get_model_type(), &addr);

    axum::serve(listener, router).await?;

    Ok(())
}

async fn run_http(hostname: String, port: String) -> Result<()> {
    let addr = format!("{}:{}", &hostname, &port);

    let router = axum::Router::new()
        .route("/health", axum::routing::get(|| async { "OK" }))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .into_make_service_with_connect_info::<std::net::SocketAddr>();

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    tracing::info!("Running {:?} HTTP server on {}", get_model_type(), &addr);

    axum::serve(listener, router).await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")), // default to "info" level
        )
        .with_target(false) // hide module path
        .compact() // short, pretty output
        .init();

    let cli = encoderfile::cli::Cli::parse();

    match cli.command {
        Commands::Serve {
            grpc_hostname,
            grpc_port,
            http_hostname,
            http_port,
            disable_grpc,
            disable_http,
        } => {
            if disable_grpc && disable_http {
                return Err(ApiError::ConfigError("Cannot disable both gRPC and HTTP"))?;
            }

            let grpc_process = match disable_grpc {
                true => tokio::spawn(async { Ok(()) }),
                false => tokio::spawn(run_grpc(grpc_hostname, grpc_port)),
            };

            let http_process = match disable_http {
                true => tokio::spawn(async { Ok(()) }),
                false => tokio::spawn(run_http(http_hostname, http_port)),
            };

            println!("{}", encoderfile::get_banner());

            let _ = tokio::join!(grpc_process, http_process);
        }
    }

    Ok(())
}
