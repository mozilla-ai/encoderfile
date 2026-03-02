use std::path::PathBuf;

use encoderfile::builder::builder::EncoderfileBuilder;
use encoderfile::builder::cli::inspect::inspect_encoderfile;
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError},
    prelude::*,
    types::PyType,
};
use pythonize::pythonize;

#[pyclass(name = "EncoderfileBuilder")]
pub struct PyEncoderfileBuilder(EncoderfileBuilder);

#[pymethods]
impl PyEncoderfileBuilder {
    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    fn from_config(_cls: &Bound<'_, PyType>, config: PathBuf) -> PyResult<PyEncoderfileBuilder> {
        EncoderfileBuilder::from_file(&config)
            .map_err(|e| PyIOError::new_err(format!("Failed to load config file: {:?}", e)))
            .map(Self)
    }

    #[pyo3(signature = (working_dir = None, version = None, no_download = false))]
    fn build(
        &self,
        working_dir: Option<&str>,
        version: Option<String>,
        no_download: bool,
    ) -> PyResult<()> {
        // change working dir if specified
        if let Some(working_dir) = working_dir {
            let path = PathBuf::from(working_dir);
            std::env::set_current_dir(path.as_path()).map_err(|_| {
                PyIOError::new_err(format!(
                    "Failed to change working directory to {:?}",
                    path.as_path()
                ))
            })?;
        }

        self.0
            .build(&version, no_download)
            .map_err(|e| PyRuntimeError::new_err(format!("Error building encoderfile: {:?}", e)))?;

        Ok(())
    }
}

#[pyfunction]
pub fn inspect<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    let result = inspect_encoderfile(&path.to_string()).map_err(|e| PyRuntimeError::new_err(format!("Failed to inspect encoderfile: {:?}", e)))?;
    pythonize(py, &result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert to Python dict: {:?}", e)))
}
