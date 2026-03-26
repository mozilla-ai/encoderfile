use pyo3::prelude::*;

mod builder;
mod cli;

/// A Python module implemented in Rust.
#[pymodule(name = "_core")]
mod encoderfile {
    #[pymodule_export]
    use super::builder::PyTargetSpec;

    #[pymodule_export]
    use super::builder::PyEncoderfileBuilder;

    #[pymodule_export]
    use super::builder::inspect;

    #[pymodule_export]
    use super::builder::PyEncoderfileConfig;

    #[pymodule_export]
    use super::builder::PyModelConfig;

    #[pymodule_export]
    use super::builder::PyInspectInfo;

    #[pymodule_export]
    use super::builder::PyBatchLongest;

    #[pymodule_export]
    use super::builder::PyFixed;

    #[pymodule_export]
    use super::builder::PyTokenizerBuildConfig;

    #[pymodule_export]
    use super::cli::run_cli;
}
