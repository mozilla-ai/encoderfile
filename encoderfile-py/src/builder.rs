use std::collections::HashMap;
use std::path::PathBuf;

use encoderfile::builder::cli::inspect::inspect_encoderfile;
use encoderfile::builder::config::{ModelPath, BuildConfig, Transform};
use encoderfile::builder::{
    builder::EncoderfileBuilder,
    cli::inspect::InspectInfo,
    config::{
        EncoderfileConfig,
        default_validate_transform,
        DEFAULT_VERSION,
    }
};
use encoderfile::common::{Config, ModelConfig, ModelType};
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError},
    prelude::*,
    types::PyType,
};

#[pyclass(name = "Transform", frozen)]
pub struct PyTransform(Transform);

#[pymethods]
impl PyTransform {
    #[staticmethod]
    fn path(path: String) -> Self {
        PyTransform(Transform::Path{path: path.into()})
    }
    #[staticmethod]
    fn inline(inline: String) -> Self {
        PyTransform(Transform::Inline(inline))
    }
    fn is_inline(&self) -> bool {
        matches!(self.0, Transform::Inline(_))
    }
    fn is_path(&self) -> bool {
        matches!(self.0, Transform::Path{..})
    }
    fn __str__(&self) -> String {
        match &self.0 {
            Transform::Path { path } => format!("Path({})", path.display()),
            Transform::Inline(inline) => format!("Inline({})", inline),
        }
    }
}

#[pyclass(name = "EncoderfileBuilder", frozen)]
pub struct PyEncoderfileBuilder(EncoderfileBuilder);

#[pymethods]
impl PyEncoderfileBuilder {
    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    fn from_configpath(_cls: &Bound<'_, PyType>, config: PathBuf) -> PyResult<PyEncoderfileBuilder> {
        EncoderfileBuilder::from_file(&config)
            .map_err(|e| PyIOError::new_err(format!("Failed to load config file: {:?}", e)))
            .map(Self)
    }

    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    #[pyo3(signature = (*, name, version = Some("0.1.0"), model_type, path, output_path = None, cache_dir = None, base_binary_path = None, transform = None, lua_libs = None, validate_transform = true, target = None))]
    fn from_config(
        _cls: &Bound<'_, PyType>, 
        name: String,
        version: Option<&str>,
        model_type: String,
        path: String,
        output_path: Option<String>,
        cache_dir: Option<String>,
        base_binary_path: Option<String>,
        transform: Option<String>,
        lua_libs: Option<Vec<String>>,
        // not yet supported
        // tokenizer: Option<String>,
        validate_transform: bool,
        target: Option<String>,
    ) -> PyResult<Self> {
        let encoderfile = EncoderfileConfig {
            name,
            version: version.unwrap_or(DEFAULT_VERSION).to_string(),
            model_type: model_type.try_into().map_err(|_| PyRuntimeError::new_err("Invalid model type"))?,
            path: ModelPath::Directory(path.into()),
            output_path: output_path.map(PathBuf::from),
            cache_dir: cache_dir.map(PathBuf::from),
            base_binary_path: base_binary_path.map(PathBuf::from),
            transform: transform.map(Transform::Inline),
            lua_libs: lua_libs.map(|libs| libs.into_iter().collect()),
            tokenizer: None,
            validate_transform,
            target,
        };
        Ok(PyEncoderfileBuilder(EncoderfileBuilder{config: BuildConfig{encoderfile}}))
    }

    #[pyo3(signature = (working_dir = None, version = None, no_download = false))]
    fn build(
        &self,
        working_dir: Option<String>,
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
