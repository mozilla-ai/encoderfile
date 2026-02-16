use anyhow::{Context, Result};
use encoderfile::build_cli::cli::GlobalArguments;
use encoderfile::build_cli::config::CONFIG_FILE_NOT_FOUND_MSG;
use std::path::Path;
use tempfile::tempdir;

#[tokio::test]
async fn test_config_does_not_exist() -> Result<()> {
    let dir = tempdir()?;
    let path = dir
        .path()
        .canonicalize()
        .expect("Failed to canonicalize temp path")
        .join("encoderfile_does_not_exist.yml");

    let build_args = encoderfile::build_cli::cli::test_build_args(
        path.as_path(),
        Path::new("dummy_binary_path"),
    );

    let global_args = GlobalArguments::default();

    let build_result = build_args
        .run(&global_args)
        .context("Failed to build encoderfile");
    assert!(build_result.is_err());
    build_result.expect_err(CONFIG_FILE_NOT_FOUND_MSG);

    Ok(())
}
