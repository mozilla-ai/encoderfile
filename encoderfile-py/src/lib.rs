use pyo3::prelude::*;

mod builder;

/// A Python module implemented in Rust.
#[pymodule]
mod encoderfile {
    #[pymodule_export]
    use super::builder::EncoderfileBuilder;
}
