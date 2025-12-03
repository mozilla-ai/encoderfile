use super::utils;
use anyhow::{Context, Result, bail};
use encoderfile_core::transforms::Transform;

pub fn validate_transform(transform: Transform) -> Result<()> {
    let batch_size = 32;
    let seq_len = 128;
    let hidden_dim = 384;

    // create dummy hidden states with shape [batch_size, seq_len, hidden_dim]
    let dummy_hidden_states =
        utils::random_tensor(&[batch_size, seq_len, hidden_dim], (-1.0, 1.0))?;

    let res = transform.postprocess(dummy_hidden_states)
        .with_context(|| "Failed to run postprocessing on dummy embeddings (randomly generated in range -1.0..1.0) of shape [32, 128, 384].")?;

    // result must return tensor of rank 3.
    if res.ndim() != 3 {
        bail!(
            "Transform must return tensor of rank 3. Got tensor of shape {:?}",
            res.shape()
        );
    }

    // result must have same batch_size and seq_len
    if res.shape()[0] != batch_size || res.shape()[1] != seq_len {
        bail!(
            "Transform must preserve batch and seq dims [{} {}, *]. Got shape {:?}",
            batch_size,
            seq_len,
            res.shape()
        );
    }

    if res.shape()[2] < 1 {
        bail!(
            "Transform returned a tensor with last dimension 0. Shape: {:?}",
            res.shape()
        );
    }

    Ok(())
}
