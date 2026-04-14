// ./encoderfile run --onnx-graph-optimization-level 1

use std::path::Path;

use anyhow::{Result, bail};
use ort::{
    execution_providers::ExecutionProvider,
    session::{
        Session,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
};

#[cfg(not(feature = "metal"))]
const METAL_ENABLED: bool = false;

#[cfg(feature = "metal")]
const METAL_ENABLED: bool = true;

#[cfg(not(feature = "cuda"))]
const CUDA_ENABLED: bool = false;

#[cfg(feature = "cuda")]
const CUDA_ENABLED: bool = true;

#[derive(Debug)]
pub struct ORTSessionBuilder {
    graph_optimization_level: Option<GraphOptimizationLevel>,
    execution_provider: Option<ORTExecutionProvider>,
}

impl ORTSessionBuilder {
    fn builder(self) -> Result<SessionBuilder> {
        let mut builder = Session::builder().map_err(|e| anyhow::anyhow!(e))?;

        if let Some(opt_level) = self.graph_optimization_level {
            builder = builder
                .with_optimization_level(opt_level)
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        match self
            .execution_provider
            .unwrap_or(ORTExecutionProvider::default())
        {
            ORTExecutionProvider::Cpu => {}
            ORTExecutionProvider::Cuda => {
                if !CUDA_ENABLED {
                    // TODO: Make this error msg better
                    bail!("This encoderfile runtime does not support CUDA.")
                }

                if ort::execution_providers::CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    bail!("CUDA unavailable.")
                }
            }
            ORTExecutionProvider::Metal => {
                if !METAL_ENABLED {
                    // TODO: Make this error msg better
                    bail!("This encoderfile runtime does not support Apple Metal.")
                }

                if ort::execution_providers::CoreMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    bail!("Apple Metal unavailable.")
                }
            }
        }

        Ok(builder)
    }

    pub fn commit_from_memory(self, payload: &[u8]) -> Result<Session> {
        self.builder()?
            .commit_from_memory(payload)
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn commit_from_file<P: AsRef<Path>>(self, path: P) -> Result<Session> {
        self.builder()?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

#[derive(Debug, Clone, Default)]
pub enum ORTExecutionProvider {
    #[default]
    Cpu,
    Metal,
    Cuda,
}
