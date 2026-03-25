use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyfunction]
pub fn run_cli(args: Vec<String>) -> PyResult<()> {
    encoderfile::builder::cli::run_cli(args).map_err(|e| {
        PyRuntimeError::new_err(format!("{}", e))
    })
}
