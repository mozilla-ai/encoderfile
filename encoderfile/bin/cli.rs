use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let cli = encoderfile::cli::Cli::parse();

    cli.command.run()
}
