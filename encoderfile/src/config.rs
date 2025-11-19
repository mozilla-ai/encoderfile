use anyhow::{Result, bail};
use schemars::JsonSchema;
use std::{io::Read, path::PathBuf};

use super::model::ModelType;
use figment::{
    Figment,
    providers::{Format, Yaml},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tera::Context;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Config {
    pub encoderfile: EncoderfileConfig,
}

impl Config {
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
    pub transform: Option<Transform>,
    #[serde(default = "default_build")]
    pub build: bool,
}

impl EncoderfileConfig {
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
            None => default_cache_dir(),
        }
    }
    pub fn to_tera_ctx(&self) -> Result<Context> {
        let mut ctx = Context::new();

        let transform = match &self.transform {
            None => None,
            Some(s) => Some(s.transform()?),
        };

        ctx.insert("version", self.version.as_str());
        ctx.insert("model_name", self.name.as_str());
        ctx.insert("model_type", &self.model_type);
        ctx.insert("model_weights_path", &self.path.model_weights_path()?);
        ctx.insert("tokenizer_path", &self.path.tokenizer_path()?);
        ctx.insert("model_config_path", &self.path.model_config_path()?);
        ctx.insert("transform", &transform);
        ctx.insert("encoderfile_version_str", &encoderfile_core_version());

        Ok(ctx)
    }

    pub fn get_generated_dir(&self) -> PathBuf {
        let filename_hash = Sha256::digest(self.name.as_bytes());

        self.cache_dir()
            .join(format!("encoderfile-{:x}", filename_hash))
    }
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

                std::fs::File::open(path)?.read_to_string(&mut code)?;

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
    },
}

macro_rules! asset_path {
    ($var:ident, $default:expr, $err:expr) => {
        pub fn $var(&self) -> Result<PathBuf> {
            let path = match self {
                Self::Paths { $var, .. } => $var.clone(),
                Self::Directory(dir) => {
                    if !dir.is_dir() {
                        bail!("No such directory: {:?}", dir);
                    }
                    dir.join($default)
                }
            };

            if !path.try_exists()? {
                bail!("Could not locate {} at path: {:?}", $err, path);
            }

            Ok(path.canonicalize()?)
        }
    };
}

impl ModelPath {
    asset_path!(model_config_path, "config.json", "model config");
    asset_path!(tokenizer_path, "tokenizer.json", "tokenizer");
    asset_path!(model_weights_path, "model.onnx", "model weights");
}

fn default_cache_dir() -> PathBuf {
    directories::ProjectDirs::from("com", "mozilla-ai", "encoderfile")
        .expect("Cannot locate")
        .cache_dir()
        .to_path_buf()
}

fn default_version() -> String {
    "0.1.0".to_string()
}

fn default_build() -> bool {
    true
}

fn encoderfile_core_version() -> &'static str {
    env!("ENCODERFILE_CORE_DEP_STR")
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

    // Create a model dir populated with the required files
    fn create_model_dir() -> PathBuf {
        let base = create_test_dir("model");
        fs::write(base.join("config.json"), "{}").expect("Failed to create config.json");
        fs::write(base.join("tokenizer.json"), "{}").expect("Failed to create tokenizer.json");
        fs::write(base.join("model.onnx"), "onnx").expect("Failed to create model.onnx");
        base
    }

    // Clean up (best-effort, don't panic)
    fn cleanup(path: &PathBuf) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn test_get_encoderfile_core_version() {
        encoderfile_core_version();
    }

    #[test]
    fn test_modelpath_directory_valid() {
        let base = create_model_dir();
        let mp = ModelPath::Directory(base.clone());

        assert!(mp.model_config_path().unwrap().ends_with("config.json"));
        assert!(mp.tokenizer_path().unwrap().ends_with("tokenizer.json"));
        assert!(mp.model_weights_path().unwrap().ends_with("model.onnx"));

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
        let base = create_model_dir();
        let mp = ModelPath::Paths {
            model_config_path: base.join("config.json"),
            tokenizer_path: base.join("tokenizer.json"),
            model_weights_path: base.join("model.onnx"),
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
        let base = create_model_dir();

        let cfg = EncoderfileConfig {
            name: "my-cool-model".into(),
            version: "1.0".into(),
            path: ModelPath::Directory(base.clone()),
            model_type: ModelType::Embedding,
            output_path: Some(base.clone()),
            cache_dir: Some(base.clone()),
            transform: None,
            build: true,
        };

        let generated = cfg.get_generated_dir();
        assert!(generated.to_string_lossy().contains("encoderfile-"));

        cleanup(&base);
    }

    #[test]
    fn test_encoderfile_to_tera_ctx() {
        let base = create_model_dir();
        let cfg = EncoderfileConfig {
            name: "sadness".into(),
            version: "0.1.0".into(),
            path: ModelPath::Directory(base.clone()),
            model_type: ModelType::SequenceClassification,
            output_path: Some(base.clone()),
            cache_dir: Some(base.clone()),
            transform: Some(Transform::Inline("1+1".into())),
            build: true,
        };

        let ctx = cfg.to_tera_ctx().expect("Tera ctx error");

        assert_eq!(ctx.get("model_name").unwrap().as_str().unwrap(), "sadness");

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

        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.encoderfile.name, "testy");
        assert_eq!(cfg.encoderfile.version, "0.9.0");

        cleanup(&dir);
    }
}
