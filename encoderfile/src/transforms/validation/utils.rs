use anyhow::{Result, Context, bail};
use encoderfile_core::transforms::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ndarray::{ArrayD, IxDyn};

pub const ERR_HEADER: &'static str = "❌ Transform validation failed";
const SEED: u64 = 42;

pub fn random_tensor(shape: &[usize], (range_start, range_end): (f32, f32)) -> Result<ArrayD<f32>> {
    let mut rng = StdRng::seed_from_u64(SEED);

    let total = shape.iter().product();

    ArrayD::from_shape_vec(
        IxDyn(shape),
        (0..total)
            .map(|_| rng.random_range(range_start..range_end))
            .collect()
    ).with_context(|| "Failed to construct random ArrayD for dry-run validation. This shouldn't happen.")
}

pub fn validation_err_ctx(msg: &str) -> String {
    format!("{}: {}", ERR_HEADER, msg)
}

pub fn validation_err<T, E: std::fmt::Display>(msg: E) -> anyhow::Result<T> {
    bail!("{}: {}", ERR_HEADER, msg)
}
