#[macro_export]
macro_rules! embed_in_section {
    ($name:ident, $path:expr, $section:expr, Bytes) => {
        $crate::embed_in_section!(
            $name,
            [u8; include_bytes!($path).len()],
            $section,
            *include_bytes!($path)
        );
    };

    ($name:ident, $path:expr, $section:expr, String) => {
        $crate::embed_in_section!($name, &str, $section, include_str!($path));
    };

    ($name:ident, $dtype:ty, $section:expr, $res:expr) => {
        $crate::embed_in_section!($name, $dtype, concat!("__DATA,", $section), $res, "macos");
        $crate::embed_in_section!($name, $dtype, concat!(".", $section), $res, "linux");
        $crate::embed_in_section!($name, $dtype, concat!(".rdata$", $section), $res, "windows");
    };

    ($name:ident, $dtype:ty, $section:expr, $res:expr, $target_os:expr) => {
        #[cfg(target_os = $target_os)]
        #[unsafe(link_section = $section)]
        #[used]
        #[unsafe(no_mangle)]
        pub static $name: $dtype = $res;
    };
}

#[macro_export]
macro_rules! factory {
    {
        $model_weights_path:expr,
        $tokenizer_path:expr,
        $model_config_path:expr,
        $model_type:expr,
        $model_id:expr,
        $transform:expr,
    } => {
        mod assets {
            $crate::embed_in_section!(MODEL_WEIGHTS, $model_weights_path, "model_weights", Bytes);
            $crate::embed_in_section!(TOKENIZER_JSON, $tokenizer_path, "tokenizer", String);
            $crate::embed_in_section!(MODEL_CONFIG_JSON, $model_config_path, "model_config", String);

            pub const MODEL_TYPE_STR: &'static str = $model_type;
            pub const MODEL_ID: &'static str = $model_id;

            #[allow(dead_code)]
            pub const TRANSFORM: Option<&'static str> = $transform;
        }

        $crate::factory! { Config }
        $crate::factory! { Model }
        $crate::factory! { Tokenizer }
        $crate::factory! { Transform }
    };

    {Config} => {
        mod config {
            use std::sync::{OnceLock, Arc};
            use $crate::{
                runtime::ModelConfig,
                common::ModelType,
            };

            static MODEL_TYPE: OnceLock<ModelType> = OnceLock::new();
            static MODEL_CONFIG: OnceLock<Arc<ModelConfig>> = OnceLock::new();

            pub fn get_model_config() -> Arc<ModelConfig> {
                MODEL_CONFIG
                    .get_or_init(
                        || match serde_json::from_str::<ModelConfig>(super::assets::MODEL_CONFIG_JSON) {
                            Ok(c) => Arc::new(c),
                            Err(e) => panic!("FATAL: Error loading model config: {e:?}"),
                        },
                    )
                    .clone()
            }

            pub fn get_model_type() -> ModelType {
                MODEL_TYPE
                    .get_or_init(|| match super::assets::MODEL_TYPE_STR {
                        "embedding" => ModelType::Embedding,
                        "sequence_classification" => ModelType::SequenceClassification,
                        "token_classification" => ModelType::TokenClassification,
                        other => panic!("Invalid model type: {other}"),
                    })
                    .clone()
            }
        }
    };

    {Model} => {
        mod model {
            use ort::session::Session;
            use parking_lot::Mutex;
            use std::sync::{Arc, OnceLock};

            static MODEL: OnceLock<Arc<Mutex<Session>>> = OnceLock::new();

            pub fn get_model() -> Arc<Mutex<Session>> {
                let model = MODEL.get_or_init(|| {
                    match Session::builder().and_then(|s| s.commit_from_memory(&super::assets::MODEL_WEIGHTS)) {
                        Ok(model) => Arc::new(Mutex::new(model)),
                        Err(e) => panic!("FATAL: Failed to load model: {e:?}"),
                    }
                });

                model.clone()
            }
        }
    };

    {Tokenizer} => {
        mod tokenizer {
            use std::sync::{Arc, OnceLock};
            use tokenizers::tokenizer::Tokenizer;
            use super::{assets::TOKENIZER_JSON, config::get_model_config};

            static TOKENIZER: OnceLock<Arc<Tokenizer>> = OnceLock::new();

            pub fn get_tokenizer() -> Arc<Tokenizer> {
                let model_config = get_model_config();

                TOKENIZER
                    .get_or_init(|| Arc::new($crate::runtime::get_tokenizer_from_string(TOKENIZER_JSON, &model_config)))
                    .clone()
            }
        }
    };

    {Transform} => {
        use $crate::transforms::Transform;

        #[cfg(not(tarpaulin_include))]
        pub fn get_transform() -> Transform {
            if let Some(script) = super::assets::TRANSFORM {
                let engine = Transform::new(script).expect("Failed to create transform");

                return engine;
            }

            Transform::new("").expect("Failed to create transform")
        }
    }
}
