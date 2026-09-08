use crate::{
    common::{ModelConfig, model_type::ModelType, TokenizerConfig},
    format::assets::{AssetKind, AssetSource, PlannedAsset},
    generated::manifest::LuaLibs as ManifestLuaLibs,
    transforms::{TransformSpec, convert_libs},
    runtime::{
        TextInputState, ImageInputState,
        ImageConfig, ImagePreprocessing, ImageSize,
    },
};
use anyhow::{Context, Result};
use std::str::FromStr;
use crate::builder::config::EncoderfileConfig;
use prost::Message;

mod embedding;
mod image_classification;
mod sentence_embedding;
mod sequence_classification;
mod token_classification;
mod utils;

pub trait TransformValidatorExt<T>: TransformSpec {
    fn validate(
        &self,
        encoderfile_config: &EncoderfileConfig,
    ) -> Result<()> {
        // if validate_transform set to false, return
        if !encoderfile_config.validate_transform {
            return Ok(());
        }

        // fail if `Postprocess` function is not found
        // NOTE: This should be removed if we add any additional functions, e.g., a Preprocess function
        if !self.has_postprocessor() {
            utils::validation_err(
                "Could not find `Postprocess` function in provided transform. Please make sure it exists.",
            )?
        }

        let input_state = self.get_input_state(encoderfile_config)?;

        self.dry_run(&encoderfile_config.model_config()?, &input_state)
    }

    fn get_input_state(&self, encoderfile_config: &EncoderfileConfig) -> Result<T>;

    fn dry_run(&self, model_config: &ModelConfig, input_state: &T) -> Result<()>;
}

macro_rules! validate_transform {
    ($transform_type:ident, $transform_str:expr, $encoderfile_config:expr) => {
        crate::transforms::$transform_type::new(
            convert_libs($encoderfile_config.lua_libs()?.as_ref()),
            Some($transform_str.clone()),
        )
        .with_context(|| utils::validation_err_ctx("Failed to create transform"))?
        .validate($encoderfile_config)
    };
}

pub fn validate_transform<'a>(
    encoderfile_config: &'a EncoderfileConfig,
) -> Result<Option<PlannedAsset<'a>>> {
    // try to fetch transform string
    // will fail if a path to a transform does not exist
    let transform_string = match &encoderfile_config.transform {
        Some(t) => t.transform()?,
        None => return Ok(None),
    };

    let transform_str = transform_string;

    match encoderfile_config.model_type {
        ModelType::Embedding => validate_transform!(
            EmbeddingTransform,
            transform_str,
            encoderfile_config
        ),
        ModelType::SequenceClassification => validate_transform!(
            SequenceClassificationTransform,
            transform_str,
            encoderfile_config
        ),
        ModelType::TokenClassification => validate_transform!(
            TokenClassificationTransform,
            transform_str,
            encoderfile_config
        ),
        ModelType::SentenceEmbedding => validate_transform!(
            SentenceEmbeddingTransform,
            transform_str,
            encoderfile_config
        ),
        ModelType::ImageClassification => validate_transform!(
            ImageClassificationTransform,
            transform_str,
            encoderfile_config
        ),
    }?;

    let lua_libs: Option<ManifestLuaLibs> = encoderfile_config
        .lua_libs
        .clone()
        .map(|libs| ManifestLuaLibs { libs });
    let proto = crate::generated::manifest::Transform {
        transform_type: crate::generated::manifest::TransformType::Lua.into(),
        transform: transform_str,
        lua_libs,
    };

    PlannedAsset::from_asset_source(
        AssetSource::InMemory(std::borrow::Cow::Owned(proto.encode_to_vec())),
        AssetKind::Transform,
    ).map(Some)
}

// TODO check if this should be removed from dev-utils - this is a copy

// Input types
pub fn get_text_input_state(encoderfile_config: &crate::builder::config::EncoderfileConfig) -> Result<TextInputState, anyhow::Error> {
    let tokenizer_str = std::fs::read_to_string(encoderfile_config.path.tokenizer_path()?)?;
    let tokenizer = get_tokenizer_from_string(&tokenizer_str);

    Ok(TextInputState {
        tokenizer,
        model_config: encoderfile_config.model_config()?,
    })
}

pub fn get_image_input_state(encoderfile_config: &crate::builder::config::EncoderfileConfig) -> Result<ImageInputState, anyhow::Error> {
    // TODO Read into serde Values and then cast into appropriate views, instead of reading
    // from file every time
    let model_config_path = encoderfile_config.path.model_config_path()?;
    let preprocessor_config_path = encoderfile_config.path.preprocessor_config_path()?.ok_or(anyhow::Error::msg("Preprocessor config path not found"))?;
    let config_state: ImageConfig = serde_json::from_str(&std::fs::read_to_string(&model_config_path)?)?;
    let preproc_state: ImagePreprocessing = serde_json::from_str(&std::fs::read_to_string(preprocessor_config_path)?)?;
    Ok(ImageInputState {
        config: ImageConfig {
            num_channels: config_state.num_channels,
            image_size: config_state.image_size,
        },
        preprocessing: ImagePreprocessing {
            do_normalize: preproc_state.do_normalize,
            do_rescale: preproc_state.do_rescale,
            do_resize: preproc_state.do_resize,
            image_processor_type: preproc_state.image_processor_type,
            rescale_factor: preproc_state.rescale_factor,
            image_mean: preproc_state.image_mean,
            image_std: preproc_state.image_std,
            size: preproc_state.size.or(Some(ImageSize {
                width: config_state.image_size,
                height: config_state.image_size,
                shortest_edge: None,
            })),
        },
    })
}

fn get_tokenizer_from_string(s: &str) -> crate::runtime::TokenizerService {
    let tokenizer = match tokenizers::tokenizer::Tokenizer::from_str(s) {
        Ok(t) => t,
        Err(e) => panic!("FATAL: Error loading tokenizer: {e:?}"),
    };

    crate::runtime::TokenizerService::new(tokenizer, TokenizerConfig::default())
        .expect("Error loading tokenizer")
}

// ---------

#[cfg(test)]
mod tests {
    use std::print;

use crate::transforms::{DEFAULT_LIBS, EmbeddingTransform};

    use crate::builder::config::{ModelPath, Transform};

    use super::*;

    fn test_encoderfile_config() -> EncoderfileConfig {
        EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from("models/embedding")),
            model_type: ModelType::Embedding,
            cache_dir: None,
            output_path: None,
            transform: None,
            lua_libs: None,
            validate_transform: true,
            tokenizer: None,
            base_binary_path: None,
            target: None,
        }
    }

    #[test]
    fn test_empty_transform() {
        let result = EmbeddingTransform::new(DEFAULT_LIBS.to_vec(), None)
            .expect("Failed to make embedding transform")
            .validate(&test_encoderfile_config());

        assert!(result.is_err())
    }

    #[test]
    fn test_no_validation() {
        let mut config = test_encoderfile_config();
        config.validate_transform = false;

        EmbeddingTransform::new(DEFAULT_LIBS.to_vec(), None)
            .expect("Failed to make embedding transform")
            .validate(&config)
            .expect("Should be ok")
    }

    #[test]
    fn test_validate() {
        let transform_str = "function Postprocess(arr) return arr end";

        print!("-> {}", std::env::current_dir().unwrap().display());

        let encoderfile_config = EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from("../models/embedding")),
            model_type: ModelType::Embedding,
            cache_dir: None,
            output_path: None,
            transform: Some(Transform::Inline(transform_str.to_string())),
            lua_libs: None,
            validate_transform: true,
            tokenizer: None,
            base_binary_path: None,
            target: None,
        };

        validate_transform(&encoderfile_config).expect("Failed to validate");
    }

    #[test]
    fn test_validate_empty() {
        let encoderfile_config = EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from("models/embedding")),
            model_type: ModelType::Embedding,
            cache_dir: None,
            output_path: None,
            transform: None,
            lua_libs: None,
            validate_transform: true,
            tokenizer: None,
            base_binary_path: None,
            target: None,
        };

        let model_config_str = include_str!(concat!(
            "../../../../../models/",
            "embedding",
            "/config.json"
        ));

        validate_transform(&encoderfile_config).expect("Failed to validate");
    }
}
