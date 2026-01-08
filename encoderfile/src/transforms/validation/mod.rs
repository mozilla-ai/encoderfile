use anyhow::{Context, Result};
use encoderfile_core::{
    common::{ModelConfig, ModelType},
    transforms::TransformSpec,
};

use crate::config::EncoderfileConfig;

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
        encoderfile_core::transforms::$transform_type::new(Some($transform_str.clone()))
            .with_context(|| utils::validation_err_ctx("Failed to create transform"))?
            .validate($encoderfile_config, $model_config)
    };
}

pub fn validate_transform(
    encoderfile_config: &EncoderfileConfig,
    model_config: &ModelConfig,
) -> Result<Option<encoderfile_core::generated::manifest::Transform>> {
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

    Ok(Some(encoderfile_core::generated::manifest::Transform {
        transform_type: encoderfile_core::generated::manifest::TransformType::Lua.into(),
        transform: transform_str,
    }))
}

#[cfg(test)]
mod tests {
    use encoderfile_core::transforms::EmbeddingTransform;

    use crate::config::{ModelPath, Transform};

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
            validate_transform: true,
            tokenizer: None,
            base_binary_path: None,
        }
    }

    fn test_model_config() -> ModelConfig {
        let config_json = include_str!("../../../../models/embedding/config.json");

        serde_json::from_str(config_json).unwrap()
    }

    #[test]
    fn test_empty_transform() {
        let result = EmbeddingTransform::new(None)
            .expect("Failed to make embedding transform")
            .validate(&test_encoderfile_config(), &test_model_config());

        assert!(result.is_err())
    }

    #[test]
    fn test_no_validation() {
        let mut config = test_encoderfile_config();
        config.validate_transform = false;

        EmbeddingTransform::new(None)
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
            path: ModelPath::Directory(std::path::PathBuf::from("models/embedding")),
            model_type: ModelType::Embedding,
            cache_dir: None,
            output_path: None,
            transform: Some(Transform::Inline(transform_str.to_string())),
            validate_transform: true,
            tokenizer: None,
            base_binary_path: None,
        };

        let model_config_str =
            include_str!(concat!("../../../../models/", "embedding", "/config.json"));

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
            validate_transform: true,
            tokenizer: None,
            base_binary_path: None,
        };

        let model_config_str =
            include_str!(concat!("../../../../models/", "embedding", "/config.json"));

        let model_config =
            serde_json::from_str(model_config_str).expect("Failed to create model config");

        validate_transform(&encoderfile_config, &model_config).expect("Failed to validate");
    }
}
