use anyhow::Result;
use encoderfile_core::transforms::Transform;

use super::validator::TransformValidator;

#[derive(Debug)]
pub struct EmbeddingTransformValidator;

impl TransformValidator for EmbeddingTransformValidator {
    fn validate(&self, transform: Transform) -> Result<()> {
        Ok(())
    }
}
