use super::{
    TransformValidatorExt,
    utils::{BATCH_SIZE, SEQ_LEN, random_tensor, validation_err, validation_err_ctx},
};
use anyhow::{Context, Result};
use encoderfile_core::{
    common::ModelConfig,
    transforms::{Postprocessor, TokenClassificationTransform},
};

impl TransformValidatorExt for TokenClassificationTransform {
    fn dry_run(&self, model_config: &ModelConfig) -> Result<()> {
        let num_labels = match model_config.num_labels() {
            Some(n) => n,
            None => validation_err(
                "Model config does not have `num_labels`, `id2label`, or `label2id` field. Please make sure you're using a TokenClassification model.",
            )?,
        };

        let dummy_logits = random_tensor(&[BATCH_SIZE, SEQ_LEN, num_labels], (-1.0, 1.0))?;
        let shape = dummy_logits.shape().to_owned();

        let res = self.postprocess(dummy_logits)
            .with_context(|| {
                validation_err_ctx(
                    format!(
                        "Failed to run postprocessing on dummy logits (randomly generated in range -1.0..1.0) of shape {:?}",
                        shape.as_slice(),
                    )
                )
            })?;

        // result must return tensor of rank 3
        if res.ndim() != 3 {
            validation_err(format!(
                "Transform must return tensor of rank 3. Got tensor of shape {:?}.",
                res.shape()
            ))?
        }

        // result must have same shape as original
        if res.shape() != shape {
            validation_err(format!(
                "Transform must return Tensor of shape [batch_size, seq_len, num_labels]. Expected shape [{}, {}, {}], got shape {:?}",
                BATCH_SIZE,
                SEQ_LEN,
                num_labels,
                res.shape()
            ))?
        }

        Ok(())
    }
}
