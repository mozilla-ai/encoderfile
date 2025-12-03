use super::utils;
use anyhow::{Context, Result, bail};
use encoderfile_core::transforms::Transform;

const BATCH_SIZE: usize = 32;
const SEQ_LEN: usize = 128;
const HIDDEN_DIM: usize = 384;

pub fn validate_transform(transform: Transform) -> Result<()> {
    // create dummy hidden states with shape [batch_size, seq_len, hidden_dim]
    let dummy_hidden_states =
        utils::random_tensor(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], (-1.0, 1.0))?;

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
    if res.shape()[0] != BATCH_SIZE || res.shape()[1] != SEQ_LEN {
        bail!(
            "Transform must preserve batch and seq dims [{} {}, *]. Got shape {:?}",
            BATCH_SIZE,
            SEQ_LEN,
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
