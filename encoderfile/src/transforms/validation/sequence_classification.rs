use super::utils::{BATCH_SIZE, random_tensor, validation_err};
use anyhow::{Context, Result, bail};
use encoderfile_core::{common::ModelConfig, transforms::Transform};

pub fn validate_transform(transform: Transform, model_config: &ModelConfig) -> Result<()> {
    let num_labels = match model_config.num_labels() {
        Some(n) => n,
        None => validation_err("Model config does not have `num_labels`, `id2label`, or `label2id` field. Please make sure you're using a SequenceClassification model.")?
    };

    Ok(())
}
