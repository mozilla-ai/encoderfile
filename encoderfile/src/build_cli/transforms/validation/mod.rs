use crate::{
    common::{ModelConfig, ModelType},
    format::assets::{AssetKind, AssetSource, PlannedAsset},
    generated::manifest::LuaLibs as ManifestLuaLibs,
    transforms::{TransformSpec, convert_libs},
};
use anyhow::{Context, Result};

use crate::build_cli::config::EncoderfileConfig;
use prost::Message;

mod embedding;
mod sentence_embedding;
mod sequence_classification;
mod token_classification;
mod utils;

pub trait TransformValidatorExt: TransformSpec {
    fn validate(
        &self,
        encoderfile_config: &EncoderfileConfig,
        model_config: &ModelConfig,
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

        self.dry_run(model_config)
    }

    fn dry_run(&self, model_config: &ModelConfig) -> Result<()>;
}

macro_rules! validate_transform {
    ($transform_type:ident, $transform_str:expr, $encoderfile_config:expr, $model_config:expr) => {
        crate::transforms::$transform_type::new(
            convert_libs($encoderfile_config.lua_libs()?.as_ref()),
            Some($transform_str.clone()),
        )
        .with_context(|| utils::validation_err_ctx("Failed to create transform"))?
        .validate($encoderfile_config, $model_config)
    };
}

pub fn validate_transform<'a>(
    encoderfile_config: &'a EncoderfileConfig,
    model_config: &'a ModelConfig,
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
            encoderfile_config,
            model_config
        ),
        ModelType::SequenceClassification => validate_transform!(
            SequenceClassificationTransform,
            transform_str,
            encoderfile_config,
            model_config
        ),
        ModelType::TokenClassification => validate_transform!(
            TokenClassificationTransform,
            transform_str,
            encoderfile_config,
            model_config
        ),
        ModelType::SentenceEmbedding => validate_transform!(
            SentenceEmbeddingTransform,
            transform_str,
            encoderfile_config,
            model_config
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
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use crate::transforms::{DEFAULT_LIBS, EmbeddingTransform};

    use crate::build_cli::config::{ModelPath, Transform};

    use super::*;

    fn test_encoderfile_config() -> EncoderfileConfig {
        EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from(
                "models/dummy_electra_token_embeddings",
            )),
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

    fn test_model_config() -> ModelConfig {
        let config_json =
            include_str!("../../../../../models/dummy_electra_token_embeddings/config.json");

        serde_json::from_str(config_json).unwrap()
    }

    #[test]
    fn test_empty_transform() {
        let result = EmbeddingTransform::new(DEFAULT_LIBS.to_vec(), None)
            .expect("Failed to make embedding transform")
            .validate(&test_encoderfile_config(), &test_model_config());

        assert!(result.is_err())
    }

    #[test]
    fn test_no_validation() {
        let mut config = test_encoderfile_config();
        config.validate_transform = false;

        EmbeddingTransform::new(DEFAULT_LIBS.to_vec(), None)
            .expect("Failed to make embedding transform")
            .validate(&config, &test_model_config())
            .expect("Should be ok")
    }

    #[test]
    fn test_validate() {
        let transform_str = "function Postprocess(arr) return arr end";

        let encoderfile_config = EncoderfileConfig {
            name: "my-model".to_string(),
            version: "0.0.1".to_string(),
            path: ModelPath::Directory(std::path::PathBuf::from(
                "models/dummy_electra_token_embeddings",
            )),
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

        let model_config_str = include_str!(concat!(
            "../../../../../models/",
            "dummy_electra_token_embeddings",
            "/config.json"
        ));

        let model_config =
            serde_json::from_str(model_config_str).expect("Failed to create model config");

        validate_transform(&encoderfile_config, &model_config).expect("Failed to validate");
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
            "dummy_electra_token_embeddings",
            "/config.json"
        ));

        let model_config =
            serde_json::from_str(model_config_str).expect("Failed to create model config");

        validate_transform(&encoderfile_config, &model_config).expect("Failed to validate");
    }
}
