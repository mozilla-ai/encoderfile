use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cli = encoderfile::cli::Cli::parse();

    cli.command.execute().await
}

#[cfg(not(tarpaulin_include))]
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,tower_http=debug,ort=warn")), // default to "info" level
        )
        .with_target(false) // hide module path
        .compact() // short, pretty output
        .init();
}
