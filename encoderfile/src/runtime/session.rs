use std::path::Path;

use anyhow::Result;
use ort::{
    execution_providers::{
        ExecutionProvider, ExecutionProviderDispatch, coreml::CoreMLComputeUnits,
    },
    session::{
        Session,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
};

#[derive(Debug)]
pub struct ORTSessionBuilder {
    pub execution_provider: ORTExecutionProvider,
    pub enable_cpu_fallback: bool,
    pub graph_optimization_level: Option<GraphOptimizationLevel>,
}

impl ORTSessionBuilder {
    fn builder(self) -> Result<SessionBuilder> {
        let mut eps = vec![self.execution_provider.dispatch()?];

        if self.enable_cpu_fallback {
            eps.push(ORTExecutionProvider::default().dispatch()?);
        }

        let optimization_level = self
            .graph_optimization_level
            .unwrap_or(GraphOptimizationLevel::Level3);

        Session::builder()
            .and_then(|b| b.with_execution_providers(eps.as_slice()))
            .and_then(|b| b.with_optimization_level(optimization_level))
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn from_memory(self, payload: &[u8]) -> Result<Session> {
        self.builder()?
            .commit_from_memory(payload)
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn from_file(self, path: impl AsRef<Path>) -> Result<Session> {
        self.builder()?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

#[derive(Debug, Clone)]
pub enum ORTExecutionProvider {
    Cpu {
        arena_allocator: bool,
    },
    Cuda {
        device_id: Option<i32>,
    },
    TensorRT {
        device_id: Option<i32>,
    },
    Metal {
        compute_units: Option<CoreMLComputeUnits>,
    },
}

impl Default for ORTExecutionProvider {
    fn default() -> Self {
        Self::Cpu {
            arena_allocator: false,
        }
    }
}

impl ORTExecutionProvider {
    fn dispatch(&self) -> Result<ExecutionProviderDispatch> {
        match self {
            Self::Cpu { arena_allocator } => get_cpu_execution_provider(*arena_allocator),
            Self::Cuda { device_id } => get_cuda_execution_provider(device_id.unwrap_or(0)),
            Self::TensorRT { device_id } => get_tensorrt_provider(device_id.unwrap_or(0)),
            Self::Metal { compute_units } => {
                get_metal_execution_provider((*compute_units).unwrap_or(CoreMLComputeUnits::All))
            }
        }
    }
}

fn get_cpu_execution_provider(arena_allocator: bool) -> Result<ExecutionProviderDispatch> {
    let ep = ort::execution_providers::CPUExecutionProvider::default()
        .with_arena_allocator(arena_allocator);

    check_provider(&ep)?;

    Ok(ep.build())
}

fn get_tensorrt_provider(device_id: i32) -> Result<ExecutionProviderDispatch> {
    let ep =
        ort::execution_providers::TensorRTExecutionProvider::default().with_device_id(device_id);

    check_provider(&ep)?;

    Ok(ep.build())
}

fn get_cuda_execution_provider(device_id: i32) -> Result<ExecutionProviderDispatch> {
    let ep = ort::execution_providers::CUDAExecutionProvider::default().with_device_id(device_id);

    check_provider(&ep)?;

    Ok(ep.build())
}

fn get_metal_execution_provider(
    compute_units: CoreMLComputeUnits,
) -> Result<ExecutionProviderDispatch> {
    let ep = ort::execution_providers::CoreMLExecutionProvider::default()
        .with_compute_units(compute_units);

    check_provider(&ep)?;

    Ok(ep.build())
}

fn check_provider<E: ExecutionProvider + std::fmt::Debug>(provider: &E) -> Result<()> {
    if !provider.is_available().unwrap_or(false) {
        anyhow::bail!("Provider {:?} is unavailable.", provider)
    }

    Ok(())
}
