use super::{
    TransformValidatorExt,
    utils::{
        BATCH_SIZE, HIDDEN_DIM, SEQ_LEN, create_dummy_attention_mask, random_tensor,
        validation_err, validation_err_ctx,
    },
};
use anyhow::{Context, Result};
use encoderfile_core::{
    common::ModelConfig,
    transforms::{Postprocessor, SentenceEmbeddingTransform},
};
use ndarray::Ix2;

impl TransformValidatorExt for SentenceEmbeddingTransform {
    fn dry_run(&self, _model_config: &ModelConfig) -> Result<()> {
        // create dummy hidden states with shape [batch_size, seq_len, hidden_dim]
        let dummy_hidden_states = random_tensor(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], (-1.0, 1.0))?;
        let dummy_attention_mask = create_dummy_attention_mask(BATCH_SIZE, SEQ_LEN - 8, SEQ_LEN)?
            .into_dimensionality::<Ix2>()
            .unwrap();
        let shape = dummy_hidden_states.shape().to_owned();

        let res = self.postprocess((dummy_hidden_states, dummy_attention_mask))
            .with_context(|| {
                validation_err_ctx(
                    format!(
                        "Failed to run postprocessing on dummy hidden states (randomly generated in range -1.0..1.0) of shape {:?}",
                        shape.as_slice(),
                    )
                )
            })?;

        // result must return tensor of rank 2
        if res.ndim() != 2 {
            validation_err(format!(
                "Transform must return tensor of rank 3. Got tensor of shape {:?}.",
                res.shape()
            ))?
        }

        // result must have same batch_size
        if res.shape()[0] != BATCH_SIZE {
            validation_err(format!(
                "Transform must preserve batch size [{}, *]. Got shape {:?}",
                BATCH_SIZE,
                res.shape()
            ))?
        }

        if res.shape()[1] < 1 {
            validation_err(format!(
                "Transform returned a tensor with last dimension 0. Shape: {:?}",
                res.shape()
            ))?
        }

        Ok(())
    }
}
