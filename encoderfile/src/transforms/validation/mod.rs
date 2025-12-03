use anyhow::{Context, Result};
use encoderfile_core::{common::ModelConfig, transforms::Transform};

use crate::{config::EncoderfileConfig, model::ModelType};

mod embedding;
mod sequence_classification;
mod utils;

#[derive(Debug)]
pub struct TransformValidator<'a> {
    // encoderfile config
    encoderfile_config: &'a EncoderfileConfig,
    // hf ModelConfig
    model_config: &'a ModelConfig,
}

impl<'a> TransformValidator<'a> {
    pub fn new(encoderfile_config: &'a EncoderfileConfig, model_config: &'a ModelConfig) -> Self {
        Self { encoderfile_config, model_config }
    }

    pub fn validate(&self) -> Result<()> {
        // if validate_transform set to false, return
        if !self.encoderfile_config.validate_transform {
            return Ok(());
        }

        // try to fetch transform string
        // will fail if a path to a transform does not exist
        let transform_str = match &self.encoderfile_config.transform {
            Some(t) => t.transform()?,
            None => return Ok(()),
        };

        // create transform
        // will fail if lua file cannot be executed
        let transform = Transform::new(transform_str.as_str())
            .with_context(|| utils::validation_err_ctx("Failed to create transform"))?;

        // fail if `Postprocess` function is not found
        // NOTE: This should be removed if we add any additional functions, e.g., a Preprocess function
        if !transform.has_postprocessor() {
            utils::validation_err(
                "Could not find `Postprocess` function in provided transform. Please make sure it exists.",
            )?
        }

        match self.encoderfile_config.model_type {
            ModelType::Embedding => embedding::validate_transform(transform),
            ModelType::SequenceClassification => sequence_classification::validate_transform(transform, self.model_config),
            ModelType::TokenClassification => todo!(),
            ModelType::SentenceEmbedding => todo!(),
        }
    }
}
