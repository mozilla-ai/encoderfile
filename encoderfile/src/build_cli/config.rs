use crate::common::{Config as EmbeddedConfig, LuaLibs, ModelConfig, ModelType};
use anyhow::{Context, Result, bail};
use schemars::JsonSchema;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    str::FromStr,
};

use figment::{
    Figment,
    providers::{Format, Yaml},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::base_binary::TargetSpec;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildConfig {
    pub encoderfile: EncoderfileConfig,
}

impl BuildConfig {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let config = Figment::new().merge(Yaml::file(path)).extract()?;

        Ok(config)
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EncoderfileConfig {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    pub path: ModelPath,
    pub model_type: ModelType,
    pub output_path: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub base_binary_path: Option<PathBuf>,
    pub transform: Option<Transform>,
    pub lua_libs: Option<Vec<String>>,
    pub tokenizer: Option<TokenizerBuildConfig>,
    #[serde(default = "default_validate_transform")]
    pub validate_transform: bool,
    pub target: Option<String>,
}

impl EncoderfileConfig {
    pub fn target(&self) -> Result<Option<TargetSpec>> {
        self.target
            .as_ref()
            .map(|s| TargetSpec::from_str(s.as_str()))
            .transpose()
    }

    pub fn embedded_config(&self) -> Result<EmbeddedConfig> {
        let config = EmbeddedConfig {
            name: self.name.clone(),
            version: self.version.clone(),
            model_type: self.model_type.clone(),
            transform: self.transform()?,
            lua_libs: None,
        };

        Ok(config)
    }

    pub fn model_config(&self) -> Result<ModelConfig> {
        let model_config_path = self.path.model_config_path()?;

        let file = File::open(model_config_path)?;

        let reader = BufReader::new(file);

        serde_json::from_reader(reader).with_context(|| "Failed to deserialize model config")
    }

    pub fn output_path(&self) -> PathBuf {
        match &self.output_path {
            Some(p) => p.to_path_buf(),
            None => {
                println!("No output path detected. Saving to current directory...");
                std::env::current_dir()
                    .expect("Can't even find the current dir? Tragic. (no seriously please open an issue)")
                    .join(format!("{}.encoderfile", self.name))
            }
        }
    }

    pub fn cache_dir(&self) -> PathBuf {
        match &self.cache_dir {
            Some(c) => c.to_path_buf(),
            None => super::cache::default_cache_dir(),
        }
    }

    pub fn transform(&self) -> Result<Option<String>> {
        let transform = match &self.transform {
            None => None,
            Some(s) => Some(s.transform()?),
        };

        Ok(transform)
    }

    pub fn lua_libs(&self) -> Result<Option<LuaLibs>> {
        let configlibs = &self.lua_libs.clone().map(LuaLibs::try_from).transpose()?;
        Ok(*configlibs)
    }

    pub fn get_generated_dir(&self) -> PathBuf {
        let filename_hash = Sha256::digest(self.name.as_bytes());

        self.cache_dir()
            .join(format!("encoderfile-{:x}", filename_hash))
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TokenizerBuildConfig {
    pub pad_strategy: Option<TokenizerPadStrategy>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged, rename_all = "snake_case")]
pub enum TokenizerPadStrategy {
    BatchLongest,
    Fixed { fixed: usize },
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Transform {
    Path { path: PathBuf },
    Inline(String),
}

impl Transform {
    pub fn transform(&self) -> Result<String> {
        match self {
            Self::Path { path } => {
                if !path.exists() {
                    bail!("No such file: {:?}", &path);
                }

                let mut code = String::new();

                File::open(path)?.read_to_string(&mut code)?;

                Ok(code)
            }
            Self::Inline(s) => Ok(s.clone()),
        }
        .map(|i| i.trim().to_string())
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(untagged)]
pub enum ModelPath {
    Directory(PathBuf),
    Paths {
        model_config_path: PathBuf,
        model_weights_path: PathBuf,
        tokenizer_path: PathBuf,
        tokenizer_config_path: Option<PathBuf>,
    },
}

impl ModelPath {
    fn resolve(
        &self,
        explicit: Option<PathBuf>,
        default: impl FnOnce(&PathBuf) -> PathBuf,
        err: &str,
    ) -> Result<Option<PathBuf>> {
        let path = match self {
            Self::Paths { .. } => explicit,
            Self::Directory(dir) => {
                if !dir.is_dir() {
                    bail!("No such directory: {:?}", dir);
                }
                Some(default(dir))
            }
        };

        match path {
            Some(p) => {
                if !p.try_exists()? {
                    bail!("Could not locate {} at path: {:?}", err, p);
                }
                Ok(Some(p.canonicalize()?))
            }
            None => Ok(None),
        }
    }
}

macro_rules! asset_path {
    (@Optional $name:ident, $default:expr, $err:expr) => {
        pub fn $name(&self) -> Result<Option<PathBuf>> {
            let explicit = match self {
                Self::Paths { $name, .. } => $name.clone(),
                _ => None,
            };

            self.resolve(explicit, |dir| dir.join($default), $err)
        }
    };

    ($name:ident, $default:expr, $err:expr) => {
        pub fn $name(&self) -> Result<PathBuf> {
            let explicit = match self {
                Self::Paths { $name, .. } => Some($name.clone()),
                _ => None,
            };

            self.resolve(explicit, |dir| dir.join($default), $err)?
                .ok_or_else(|| anyhow::anyhow!("Missing required path: {}", $err))
        }
    };
}

impl ModelPath {
    asset_path!(model_config_path, "config.json", "model config");
    asset_path!(tokenizer_path, "tokenizer.json", "tokenizer");
    asset_path!(model_weights_path, "model.onnx", "model weights");
    asset_path!(@Optional tokenizer_config_path, "tokenizer_config.json", "tokenizer config");
}

fn default_version() -> String {
    "0.1.0".to_string()
}

fn default_validate_transform() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::PathBuf};

    // Create a stable, normal directory under the system temp dir
    fn create_test_dir(name: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "encoderfile-test-{}-{}",
            name,
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&base).unwrap();
        base
    }

    // Create temp output dir
    fn create_temp_output_dir() -> PathBuf {
        create_test_dir("model")
    }

    // Create a model dir populated with the required files
    fn create_temp_model_dir() -> PathBuf {
        let base = create_test_dir("model");
        fs::write(base.join("config.json"), "{}").expect("Failed to create config.json");
        fs::write(base.join("tokenizer.json"), "{}").expect("Failed to create tokenizer.json");
        fs::write(base.join("model.onnx"), "onnx").expect("Failed to create model.onnx");
        fs::write(base.join("tokenizer_config.json"), "{}")
            .expect("Failed to create tokenizer_config.json");
        base
    }

    // Clean up (best-effort, don't panic)
    fn cleanup(path: &PathBuf) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn test_modelpath_directory_valid() {
        let base = create_temp_model_dir();
        let mp = ModelPath::Directory(base.clone());

        assert!(mp.model_config_path().unwrap().ends_with("config.json"));
        assert!(mp.tokenizer_path().unwrap().ends_with("tokenizer.json"));
        assert!(mp.model_weights_path().unwrap().ends_with("model.onnx"));
        assert!(
            mp.tokenizer_config_path()
                .unwrap()
                .unwrap()
                .ends_with("tokenizer_config.json")
        );

        cleanup(&base);
    }

    #[test]
    fn test_modelpath_directory_missing_file() {
        let base = create_test_dir("missing");
        let mp = ModelPath::Directory(base.clone());

        let err = mp.model_config_path().unwrap_err();
        assert!(err.to_string().contains("model config"));

        cleanup(&base);
    }

    #[test]
    fn test_modelpath_explicit_paths() {
        let base = create_temp_model_dir();
        let mp = ModelPath::Paths {
            model_config_path: base.join("config.json"),
            tokenizer_path: base.join("tokenizer.json"),
            model_weights_path: base.join("model.onnx"),
            tokenizer_config_path: Some(base.join("tokenizer_config.json")),
        };

        assert!(mp.model_config_path().is_ok());

        cleanup(&base);
    }

    #[test]
    fn test_transform_inline() {
        let t = Transform::Inline("  hello world   ".into());
        assert_eq!(t.transform().unwrap(), "hello world");
    }

    #[test]
    fn test_transform_path() {
        let dir = create_test_dir("transform");
        let file = dir.join("script.txt");

        fs::write(&file, "   goodbye world ").unwrap();

        let t = Transform::Path { path: file };
        assert_eq!(t.transform().unwrap(), "goodbye world");

        cleanup(&dir);
    }

    #[test]
    fn test_transform_missing_file() {
        let bogus = PathBuf::from("totally-does-not-exist.txt");
        let t = Transform::Path {
            path: bogus.clone(),
        };

        let err = t.transform().unwrap_err();
        assert!(err.to_string().contains("No such file"));
    }

    #[test]
    fn test_encoderfile_generated_dir() {
        let base = create_temp_output_dir();

        let cfg = EncoderfileConfig {
            name: "my-cool-model".into(),
            version: "1.0".into(),
            path: ModelPath::Directory("../models/embedding".into()),
            model_type: ModelType::Embedding,
            output_path: Some(base.clone()),
            cache_dir: Some(base.clone()),
            validate_transform: false,
            transform: None,
            lua_libs: None,
            tokenizer: None,
            base_binary_path: None,
            target: None,
        };

        let generated = cfg.get_generated_dir();
        assert!(generated.to_string_lossy().contains("encoderfile-"));

        cleanup(&base);
    }

    #[test]
    fn test_config_loading() {
        let dir = create_test_dir("config");
        let path = dir.join("config.yml");

        let yaml = r#"
encoderfile:
  name: testy
  version: "0.9.0"
  path: "./"
  model_type: embedding
"#;

        fs::write(&path, yaml).unwrap();

        let cfg = BuildConfig::load(&path).unwrap();
        assert_eq!(cfg.encoderfile.name, "testy");
        assert_eq!(cfg.encoderfile.version, "0.9.0");

        cleanup(&dir);
    }
}
