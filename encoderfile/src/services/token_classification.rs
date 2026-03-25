use crate::{
    common::{TokenClassificationRequest, TokenClassificationResponse, model_type},
    error::ApiError,
    inference,
    runtime::AppState,
    transforms::TokenClassificationTransform,
};

use ort::execution_providers::{self as ep, ExecutionProvider};

use super::inference::Inference;

impl Inference for AppState<model_type::TokenClassification> {
    type Input = TokenClassificationRequest;
    type Output = TokenClassificationResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();

        let session = self.session.lock();
        tracing::info!("WAIT WOT?");
        tracing::info!("TensorRT? {:?}", ep::TensorRTExecutionProvider::default().is_available());
        tracing::info!("CUDA? {:?}", ep::CUDAExecutionProvider::default().is_available());
        tracing::info!("DirectML? {:?}", ep::DirectMLExecutionProvider::default().is_available());
        tracing::info!("CoreML? {:?}", ep::CoreMLExecutionProvider::default().is_available());
        let encodings = self.tokenizer.encode_text(request.inputs)?;

        let transform =
            TokenClassificationTransform::new(self.lua_libs.clone(), self.transform_str())?;

        let results = inference::token_classification::token_classification(
            session,
            &transform,
            &self.model_config,
            encodings,
        )?;

        Ok(TokenClassificationResponse {
            results,
            model_id: self.config.name.clone(),
            metadata: request.metadata,
        })
    }
}
