use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let cli = encoderfile::builder::cli::Cli::parse();

    cli.command.run(&cli.global_args)
}
