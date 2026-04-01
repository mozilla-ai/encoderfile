use std::collections::HashMap;
use std::path::PathBuf;

use encoderfile::builder::base_binary::TargetSpec;
use encoderfile::builder::cli::inspect::inspect_encoderfile;
use encoderfile::builder::config::{BuildConfig, ModelPath, TokenizerBuildConfig, Transform};
use encoderfile::builder::{
    builder::EncoderfileBuilder,
    cli::inspect::InspectInfo,
    config::{
        DEFAULT_VERSION, EncoderfileConfig, TokenizerPadStrategy, TokenizerTruncationSide,
        TokenizerTruncationStrategy,
    },
};
use encoderfile::common::{Config, ModelConfig};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyString;
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError, PyValueError},
    prelude::*,
};

#[pyclass(name = "BatchLongest")]
#[derive(Debug, Clone)]
pub struct PyBatchLongest;

#[pymethods]
impl PyBatchLongest {
    #[new]
    pub fn new() -> Self {
        PyBatchLongest
    }
}

impl Default for PyBatchLongest {
    fn default() -> Self {
        PyBatchLongest
    }
}

impl From<PyBatchLongest> for TokenizerPadStrategy {
    fn from(_value: PyBatchLongest) -> Self {
        TokenizerPadStrategy::BatchLongest
    }
}

#[pyclass(name = "Fixed")]
#[derive(Debug, Clone)]
pub struct PyFixed {
    #[pyo3(get, set)]
    n: usize,
}

#[pymethods]
impl PyFixed {
    #[new]
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl From<PyFixed> for TokenizerPadStrategy {
    fn from(value: PyFixed) -> Self {
        TokenizerPadStrategy::Fixed { fixed: value.n }
    }
}

#[pyclass(name = "TokenizerBuildConfig", frozen)]
pub struct PyTokenizerBuildConfig(TokenizerBuildConfig);

#[pymethods]
impl PyTokenizerBuildConfig {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, pad_strategy = None, truncation_side = None, truncation_strategy = None, max_length = None, stride = None))]
    pub fn new(
        pad_strategy: Option<Bound<'_, PyAny>>,
        truncation_side: Option<String>,
        truncation_strategy: Option<String>,
        max_length: Option<usize>,
        stride: Option<usize>,
    ) -> PyResult<Self> {
        let pad_strategy = match pad_strategy {
            Some(ps) => {
                if let Ok(batch_longest) = ps.cast::<PyBatchLongest>() {
                    Some(batch_longest.extract::<PyBatchLongest>().unwrap().into())
                } else if let Ok(fixed) = ps.cast::<PyFixed>() {
                    Some(fixed.extract::<PyFixed>().unwrap().into())
                } else {
                    return Err(PyTypeError::new_err(
                        "Class must be BatchLongest, Fixed, or None",
                    ));
                }
            }
            None => None,
        };

        let truncation_side = truncation_side
            .map(|s| {
                s.parse::<TokenizerTruncationSide>()
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse: {:?}", e)))
            })
            .transpose()?;
        let truncation_strategy = truncation_strategy
            .map(|s| {
                s.parse::<TokenizerTruncationStrategy>()
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse: {:?}", e)))
            })
            .transpose()?;

        Ok(PyTokenizerBuildConfig(TokenizerBuildConfig {
            pad_strategy,
            truncation_side,
            truncation_strategy,
            max_length,
            stride,
        }))
    }

    #[getter]
    pub fn get_pad_strategy(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match self.0.pad_strategy.as_ref() {
            Some(ps) => match ps {
                TokenizerPadStrategy::BatchLongest => Some(PyBatchLongest::new().into_py_any(py)),
                TokenizerPadStrategy::Fixed { fixed } => Some(PyFixed::new(*fixed).into_py_any(py)),
            },
            None => None,
        }
        .transpose()
    }

    #[getter]
    pub fn get_truncation_side(&self) -> Option<String> {
        self.0.truncation_side.as_ref().map(|s| s.into())
    }

    #[getter]
    pub fn get_truncation_strategy(&self) -> Option<String> {
        self.0.truncation_strategy.as_ref().map(|s| s.into())
    }

    #[getter]
    pub fn get_max_length(&self) -> Option<usize> {
        self.0.max_length
    }

    #[getter]
    pub fn get_stride(&self) -> Option<usize> {
        self.0.stride
    }
}

#[pyclass(name = "EncoderfileBuilder", frozen)]
pub struct PyEncoderfileBuilder(EncoderfileBuilder);

#[pymethods]
impl PyEncoderfileBuilder {
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    fn from_config(config_path: PathBuf) -> PyResult<PyEncoderfileBuilder> {
        EncoderfileBuilder::from_file(&config_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to load config file: {:?}", e)))
            .map(Self)
    }

    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, name, version = None, model_type, path, output_path = None, cache_dir = None, base_binary_path = None, transform = None, lua_libs = None, tokenizer = None, validate_transform = true, target = None))]
    fn from_dict(
        name: String,
        version: Option<&str>,
        model_type: String,
        path: String,
        output_path: Option<String>,
        cache_dir: Option<String>,
        base_binary_path: Option<String>,
        transform: Option<String>,
        lua_libs: Option<Vec<String>>,
        tokenizer: Option<Bound<'_, PyTokenizerBuildConfig>>,
        validate_transform: bool,
        target: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        let encoderfile = EncoderfileConfig {
            name,
            version: version.unwrap_or(DEFAULT_VERSION).to_string(),
            model_type: model_type
                .parse()
                .map_err(|_| PyRuntimeError::new_err("Invalid model type"))?,
            path: ModelPath::Directory(path.into()),
            output_path: output_path.map(PathBuf::from),
            cache_dir: cache_dir.map(PathBuf::from),
            base_binary_path: base_binary_path.map(PathBuf::from),
            transform: transform.map(Transform::Inline),
            lua_libs: lua_libs.map(|libs| libs.into_iter().collect()),
            tokenizer: tokenizer.map(|t| t.borrow().0.clone()),
            validate_transform,
            target: target
                .map(|t| {
                    if let Ok(spec) = t.cast::<PyString>() {
                        PyTargetSpec::parse(spec.to_str()?)
                    } else if let Ok(spec) = t.cast::<PyTargetSpec>() {
                        Ok(PyTargetSpec(spec.get().0.clone()))
                    } else {
                        Err(PyRuntimeError::new_err(
                            "Failed to parse target spec: expected either a string or a TargetSpec",
                        ))
                    }
                })
                .transpose()?
                .map(|t| t.0),
        };
        Ok(PyEncoderfileBuilder(EncoderfileBuilder {
            config: BuildConfig { encoderfile },
        }))
    }

    #[pyo3(signature = (workdir = None, version = None, no_download = false))]
    fn build(
        &self,
        workdir: Option<String>,
        version: Option<String>,
        no_download: bool,
    ) -> PyResult<()> {
        // change working dir if specified
        if let Some(workdir) = workdir {
            let path = PathBuf::from(workdir);
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
        self.0.model_type.to_string()
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

#[pyclass(name = "TargetSpec", frozen)]
#[derive(Clone)]
pub struct PyTargetSpec(TargetSpec);

#[pymethods]
impl PyTargetSpec {
    #[new]
    #[pyo3(signature = (spec))]
    fn parse(spec: &str) -> PyResult<Self> {
        spec.parse()
            .map(PyTargetSpec)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse target spec: {:?}", e)))
    }

    #[getter]
    fn get_arch(&self) -> String {
        self.0.arch.to_string()
    }

    #[getter]
    fn get_os(&self) -> String {
        self.0.os.to_string()
    }

    #[getter]
    fn get_abi(&self) -> String {
        self.0.abi.to_string()
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

#[pyfunction]
pub fn read_metadata(_py: Python<'_>, path: &str) -> PyResult<PyInspectInfo> {
    let result = inspect_encoderfile(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to inspect encoderfile: {:?}", e)))?;
    Ok(PyInspectInfo(result))
}
