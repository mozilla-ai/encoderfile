use std::{fs::File, io::BufReader};

use anyhow::Result;
use clap::Parser;
use encoderfile::{runtime::load_assets, transport::cli::Cli};

#[tokio::main]
async fn main() -> Result<()> {
    // parse CLI
    let cli = Cli::parse();

    // open current executable
    let path = std::env::current_exe()?;
    let file = File::open(path)?;
    let mut file = BufReader::new(file);

    // load encoderfile
    let mut loader = load_assets(&mut file)?;

    // execute
    cli.command.execute(&mut loader).await
}
