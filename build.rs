use schemars::{schema_for, JsonSchema};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Embedding,
    SequenceClassification,
    TokenClassification
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BuildConfig {
    pub model: ModelConfig,
}

macro_rules! alt_path {
    ($path:expr, $alt_path:expr, $alt_path_name:expr, $default:expr) => {{
        if $path.is_none() && $alt_path.is_none() {
            return Err(format!(
                "Either path OR {} must be specified.",
                $alt_path_name
            ));
        }

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map_err(|_| "CARGO_MANIFEST_DIR not set".to_string())?;

        let raw_path = match &$path {
            Some(path) => std::path::Path::new(path).join($default),
            None => match &$alt_path {
                Some(path) => std::path::Path::new(path).to_path_buf(),
                None => {
                    return Err(format!(
                        "Either path OR {} must be specified.",
                        $alt_path_name
                    ));
                }
            },
        };

        // If path is relative, resolve relative to the crate root
        let file_path = if raw_path.is_absolute() {
            raw_path
        } else {
            std::path::Path::new(&manifest_dir).join(&raw_path)
        };

        if !file_path.is_file() {
            return Err(format!("Cannot find {}", file_path.to_string_lossy()));
        }

        Ok(file_path.to_string_lossy().to_string())
    }};
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ModelConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub path: Option<String>,
    model_weights_path: Option<String>,
    model_config_path: Option<String>,
    tokenizer_path: Option<String>,
}

impl ModelConfig {
    pub fn model_weights_path(&self) -> Result<String, String> {
        alt_path!(
            self.path,
            self.model_weights_path,
            "model_weights_path",
            "model.onnx"
        )
    }

    pub fn model_config_path(&self) -> Result<String, String> {
        alt_path!(
            self.path,
            self.model_config_path,
            "model_config_path",
            "config.json"
        )
    }

    pub fn tokenizer_path(&self) -> Result<String, String> {
        alt_path!(
            self.path,
            self.tokenizer_path,
            "tokenizer_path",
            "tokenizer.json"
        )
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    let schema = schema_for!(BuildConfig);
    std::fs::write("configs/build_config.schema.json", serde_json::to_string_pretty(&schema)?)?;

    tonic_prost_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .build_server(true)
        .out_dir("src/generated")
        .compile_protos(&["proto/encoderfile.proto"], &["proto/encoderfile"])?;

    let config_path = std::env::var("ENCODERFILE_CONFIG_PATH").unwrap();

    let config_str = std::fs::read_to_string(config_path)?;
    let config: BuildConfig = toml::from_str(&config_str)?;

    println!(
        "cargo:rustc-env=MODEL_WEIGHTS_PATH={}",
        &config.model.model_weights_path()?
    );
    println!(
        "cargo:rustc-env=TOKENIZER_PATH={}",
        &config.model.tokenizer_path()?
    );
    println!(
        "cargo:rustc-env=MODEL_CONFIG_PATH={}",
        &config.model.model_config_path()?
    );
    println!("cargo:rustc-env=MODEL_TYPE={}", &config.model.type_);
    println!("cargo:rustc-env=MODEL_NAME={}", &config.model.name);

    Ok(())
}
