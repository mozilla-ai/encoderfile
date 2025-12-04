use anyhow::{Context, Result};
use encoderfile_core::{
    common::ModelConfig,
    transforms::{
        EmbeddingTransform, SentenceEmbeddingTransform, SequenceClassificationTransform,
        TokenClassificationTransform, TransformSpec,
    },
};

use crate::{config::EncoderfileConfig, model::ModelType};

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

pub fn validate_transform(
    encoderfile_config: &EncoderfileConfig,
    model_config: &ModelConfig,
) -> Result<()> {
    // try to fetch transform string
    // will fail if a path to a transform does not exist
    let transform_string = match &encoderfile_config.transform {
        Some(t) => t.transform()?,
        None => return Ok(()),
    };

    let transform_str = Some(transform_string.as_ref());

    match encoderfile_config.model_type {
        ModelType::Embedding => EmbeddingTransform::new(transform_str)
            .with_context(|| utils::validation_err_ctx("Failed to create transform"))?
            .validate(encoderfile_config, model_config),
        ModelType::SequenceClassification => SequenceClassificationTransform::new(transform_str)
            .with_context(|| utils::validation_err_ctx("Failed to create transform"))?
            .validate(encoderfile_config, model_config),
        ModelType::TokenClassification => TokenClassificationTransform::new(transform_str)
            .with_context(|| utils::validation_err_ctx("Failed to create transform"))?
            .validate(encoderfile_config, model_config),
        ModelType::SentenceEmbedding => SentenceEmbeddingTransform::new(transform_str)
            .with_context(|| utils::validation_err_ctx("Failed to create transform"))?
            .validate(encoderfile_config, model_config),
    }
}
