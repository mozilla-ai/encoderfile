use anyhow::Result;
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

    Ok(())
}
