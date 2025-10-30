use clap::Parser;
use anyhow::Result;
use encoderfile::cli::{Commands, ServeCommands};
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
        Commands::Serve { command } => {
            match command {
                ServeCommands::Grpc { hostname, port } => {
                    let addr = format!("{}:{}", hostname, port).parse().unwrap();

                    println!("{}", encoderfile::get_banner());

                    encoderfile::grpc::router()
                        .serve(addr)
                        .await?;
                },
                ServeCommands::Http { hostname, port } => {}
            }
        }
    };

    Ok(())
}
