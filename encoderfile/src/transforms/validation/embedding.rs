use super::utils::{BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, random_tensor, validation_err, validation_err_ctx};
use anyhow::{Context, Result};
use encoderfile_core::transforms::Transform;

pub fn validate_transform(transform: Transform) -> Result<()> {
    // create dummy hidden states with shape [batch_size, seq_len, hidden_dim]
    let dummy_hidden_states =
        random_tensor(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], (-1.0, 1.0))?;
    let shape = dummy_hidden_states.shape().to_owned();

    let res = transform.postprocess(dummy_hidden_states)
        .with_context(|| {
            validation_err_ctx(
                format!(
                    "Failed to run postprocessing on dummy logits (randomly generated in range -1.0..1.0) of shape {:?}",
                    shape.as_slice(),
                )
            )
        })?;

    // result must return tensor of rank 3.
    if res.ndim() != 3 {
        validation_err(
            format!(
                "Transform must return tensor of rank 3. Got tensor of shape {:?}.",
                res.shape()
            )
        )?
    }

    // result must have same batch_size and seq_len
    if res.shape()[0] != BATCH_SIZE || res.shape()[1] != SEQ_LEN {
        validation_err(
            format!(
                "Transform must preserve batch and seq dims [{} {}, *]. Got shape {:?}",
                BATCH_SIZE,
                SEQ_LEN,
                res.shape()
            )
        )?
    }

    if res.shape()[2] < 1 {
        validation_err(
            format!(
                "Transform returned a tensor with last dimension 0. Shape: {:?}",
                res.shape()
            )
        )?
    }

    Ok(())
}
