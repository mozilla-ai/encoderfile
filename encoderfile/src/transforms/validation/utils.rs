use anyhow::{Context, Result, bail};
use ndarray::{ArrayD, IxDyn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const ERR_HEADER: &'static str = "❌ Transform validation failed";
const SEED: u64 = 42;

// test values
pub const BATCH_SIZE: usize = 32;
pub const SEQ_LEN: usize = 128;
pub const HIDDEN_DIM: usize = 384;

pub fn random_tensor(shape: &[usize], (range_start, range_end): (f32, f32)) -> Result<ArrayD<f32>> {
    let mut rng = StdRng::seed_from_u64(SEED);

    let total = shape.iter().product();

    ArrayD::from_shape_vec(
        IxDyn(shape),
        (0..total)
            .map(|_| rng.random_range(range_start..range_end))
            .collect(),
    )
    .with_context(
        || validation_err_ctx("Failed to construct random ArrayD for dry-run validation. This shouldn't happen. More details"),
    )
}

pub fn validation_err_ctx<T: std::fmt::Display>(msg: T) -> String {
    format!("{}: {}", ERR_HEADER, msg)
}

pub fn validation_err<T, E: std::fmt::Display>(msg: E) -> Result<T> {
    bail!("{}: {}", ERR_HEADER, msg)
}
