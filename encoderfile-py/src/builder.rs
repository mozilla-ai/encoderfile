use std::path::PathBuf;

use encoderfile::builder::cli::{BuildArgs, GlobalArguments};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyType,
};

#[pyclass]
pub struct EncoderfileBuilder(BuildArgs);

#[pymethods]
impl EncoderfileBuilder {
    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    #[pyo3(signature = (
        config,
        output_path = None,
        base_binary_path = None,
        platform = None,
        version = None,
        no_download = false,
        directory = None))
        ]
    fn from_config(
        _cls: &Bound<'_, PyType>,
        config: PathBuf,
        output_path: Option<PathBuf>,
        base_binary_path: Option<PathBuf>,
        platform: Option<String>,
        version: Option<String>,
        no_download: Option<bool>,
        directory: Option<PathBuf>,
    ) -> PyResult<EncoderfileBuilder> {
        let platform = platform
            .as_ref()
            .map(|i| std::str::FromStr::from_str(i.as_str()))
            .transpose()
            .map_err(|_| PyValueError::new_err(format!("Invalid platform: {:?}", &platform)))?;

        Ok(EncoderfileBuilder(BuildArgs {
            config,
            output_path,
            base_binary_path,
            platform,
            version,
            no_download: no_download.unwrap_or(false),
            working_dir: directory,
        }))
    }

    #[pyo3(signature = (cache_dir = None))]
    fn build(&self, cache_dir: Option<PathBuf>) -> PyResult<()> {
        let global = GlobalArguments { cache_dir };

        self.0
            .run(&global)
            .map_err(|e| PyRuntimeError::new_err(format!("Error building encoderfile: {:?}", e)))?;

        Ok(())
    }
}
