use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = encoderfile::cli::Cli::parse();

    cli.command.execute().await
}
