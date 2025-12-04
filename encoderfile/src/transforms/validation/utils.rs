use anyhow::{Context, Result, bail};
use ndarray::{Array, ArrayD, Dimension, IxDyn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const ERR_HEADER: &str = "❌ Transform validation failed";
const SEED: u64 = 42;

// test values
pub const BATCH_SIZE: usize = 32;
pub const SEQ_LEN: usize = 128;
pub const HIDDEN_DIM: usize = 384;

pub fn random_tensor<D: Dimension>(
    shape: &[usize],
    (range_start, range_end): (f32, f32),
) -> Result<Array<f32, D>> {
    let mut rng = StdRng::seed_from_u64(SEED);

    let total = shape.iter().product();

    ArrayD::from_shape_vec(
        shape,
        (0..total)
            .map(|_| rng.random_range(range_start..range_end))
            .collect(),
    )
    .and_then(|i| i.into_dimensionality::<D>())
    .with_context(
        || validation_err_ctx("Failed to construct random ArrayD for dry-run validation. This shouldn't happen. More details"),
    )
}

pub fn create_dummy_attention_mask<D: Dimension>(
    batch: usize,
    seq: usize,
    pad_up_to: usize,
) -> Result<Array<f32, D>> {
    let mut data = Vec::with_capacity(batch * seq);

    for _ in 0..batch {
        let real = seq - pad_up_to;
        data.extend(std::iter::repeat_n(1.0, real));
        data.extend(std::iter::repeat_n(0.0, pad_up_to));
    }

    ArrayD::from_shape_vec(IxDyn(&[batch, seq]), data)
        .and_then(|i| i.into_dimensionality::<D>())
        .with_context(
            || validation_err_ctx("Failed to construct dummy attention_mask for dry-run validation. This shouldn't happen. More details"),
        )
}

pub fn validation_err_ctx<T: std::fmt::Display>(msg: T) -> String {
    format!("{}: {}", ERR_HEADER, msg)
}

pub fn validation_err<T, E: std::fmt::Display>(msg: E) -> Result<T> {
    bail!("{}: {}", ERR_HEADER, msg)
}
