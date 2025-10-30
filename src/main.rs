use anyhow::Result;
use clap::Parser;
use encoderfile::{
    cli::{Commands, ServeCommands},
    config::get_model_type,
};
use tracing_subscriber::EnvFilter;

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

    match &cli.command {
        Commands::Serve { command } => match command {
            ServeCommands::Grpc { hostname, port } => {
                let addr = format!("{}:{}", hostname, port);

                println!("{}", encoderfile::get_banner());

                tracing::info!(
                    "Serving {:?} model {} on gRPC {}",
                    get_model_type(),
                    encoderfile::MODEL_ID,
                    &addr
                );

                let router = encoderfile::grpc::router()
                    .layer(tower_http::trace::TraceLayer::new_for_grpc());

                let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

                axum::serve(
                    listener,
                    router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
                )
                .await?;
            }
            ServeCommands::Http { hostname, port } => {}
        },
    };

    Ok(())
}
