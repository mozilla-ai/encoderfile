use std::collections::HashMap;
use std::path::PathBuf;

use encoderfile::builder::cli::inspect::inspect_encoderfile;
use encoderfile::builder::{builder::EncoderfileBuilder, cli::inspect::InspectInfo};
use encoderfile::common::Config;
use encoderfile::common::ModelConfig;
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError},
    prelude::*,
    types::PyType,
};

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

#[pyclass(name = "InspectInfo", frozen)]
pub struct PyInspectInfo(InspectInfo);

#[pymethods]
impl PyInspectInfo {
    #[getter]
    fn get_model_config(&self) -> PyResult<PyModelConfig> {
        Ok(PyModelConfig(self.0.model_config.clone()))
    }
    #[getter]
    fn get_encoderfile_config(&self) -> PyResult<PyEncoderfileConfig> {
        Ok(PyEncoderfileConfig(self.0.encoderfile_config.clone()))
    }
}

#[pyclass(name = "ModelConfig", frozen)]
pub struct PyModelConfig(ModelConfig);

#[pymethods]
impl PyModelConfig {
    #[getter]
    fn get_model_type(&self) -> String {
        self.0.model_type.clone()
    }

    #[getter]
    fn get_num_labels(&self) -> Option<usize> {
        self.0.num_labels()
    }

    #[getter]
    fn get_id2label(&self) -> Option<HashMap<u32, String>> {
        self.0.id2label.clone()
    }

    #[getter]
    fn get_label2id(&self) -> Option<HashMap<String, u32>> {
        self.0.label2id.clone()
    }
}

#[pyclass(name = "EncoderfileConfig", frozen)]
pub struct PyEncoderfileConfig(Config);

#[pymethods]
impl PyEncoderfileConfig {
    #[getter]
    fn get_name(&self) -> String {
        self.0.name.clone()
    }

    #[getter]
    fn get_version(&self) -> String {
        self.0.version.clone()
    }

    #[getter]
    fn get_model_type(&self) -> String {
        format!("{:?}", self.0.model_type)
    }

    #[getter]
    fn get_transform(&self) -> Option<String> {
        self.0.transform.clone()
    }

    #[getter]
    fn get_lua_libs(&self) -> Option<Vec<String>> {
        Some(self.0.lua_libs?.into())
    }
}

#[pyfunction]
pub fn inspect(_py: Python<'_>, path: &str) -> PyResult<PyInspectInfo> {
    let result = inspect_encoderfile(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to inspect encoderfile: {:?}", e)))?;
    Ok(PyInspectInfo(result))
}
