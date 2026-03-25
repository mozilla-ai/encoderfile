use pyo3::{exceptions::PyRuntimeError, prelude::*};

// For use ONLY in encoderfile-py's __main__.py
#[pyfunction]
pub fn run_cli(args: Vec<String>) -> PyResult<()> {
    encoderfile::builder::cli::run_cli(args).map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
}
