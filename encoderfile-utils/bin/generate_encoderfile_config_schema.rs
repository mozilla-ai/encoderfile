use anyhow::Result;
use encoderfile::config;

fn main() -> Result<()> {
    let schema = schemars::schema_for!(config::BuildConfig);

    let schema_str = serde_json::to_string_pretty(&schema)?;

    let out_path = std::path::PathBuf::from("schemas").join("encoderfile-config-schema.json");

    std::fs::write(out_path, schema_str)?;

    Ok(())
}
