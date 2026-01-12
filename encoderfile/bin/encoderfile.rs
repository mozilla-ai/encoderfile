use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let cli = encoderfile::build_cli::cli::Cli::parse();

    cli.command.run(&cli.global_args)
}
