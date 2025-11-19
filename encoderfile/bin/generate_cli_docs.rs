use anyhow::Result;
use clap_markdown::help_markdown;
use encoderfile::cli::Cli;

fn main() -> Result<()> {
    let markdown = help_markdown::<Cli>();
    std::fs::write("docs/reference/encoderfile_util_cli.md", markdown)?;
    Ok(())
}
