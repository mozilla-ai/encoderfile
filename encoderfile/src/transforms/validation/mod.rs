use anyhow::{Context, Result};
use encoderfile_core::transforms::Transform;

use crate::{config::EncoderfileConfig, model::ModelType};

mod utils;
mod embedding;

#[derive(Debug)]
pub struct TransformValidator {
    config: EncoderfileConfig
}

impl TransformValidator {
    pub fn new(config: EncoderfileConfig) -> Self {
        Self { config }
    }

    pub fn validate(&self) -> Result<()> {
        // if validate_transform set to false, return
        if !self.config.validate_transform {
            return Ok(())
        }

        // try to fetch transform string
        // will fail if a path to a transform does not exist
        let transform_str = match &self.config.transform {
            Some(t) => t.transform()?,
            None => return Ok(())
        };

        // create transform
        // will fail if lua file cannot be executed
        let transform = Transform::new(transform_str.as_str())
            .with_context(|| utils::validation_err_ctx("Failed to create transform"))?;

        // fail if `Postprocess` function is not found
        // NOTE: This should be removed if we add any additional functions, e.g., a Preprocess function
        if !transform.has_postprocessor() {
            utils::validation_err("❌ Could not find `Postprocess` function in provided transform. Please make sure it exists.")?
        }

        match self.config.model_type {
            ModelType::Embedding => embedding::validate_transform(transform),
            ModelType::SequenceClassification => todo!(),
            ModelType::TokenClassification => todo!(),
            ModelType::SentenceEmbedding => todo!(),
        }
    }
}
