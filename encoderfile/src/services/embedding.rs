use crate::{
    common::{EmbeddingRequest, EmbeddingResponse, model_type},
    error::ApiError,
    inference,
    runtime::AppState,
    transforms::EmbeddingTransform,
};

use super::inference::Inference;

impl Inference for AppState<model_type::Embedding> {
    type Input = EmbeddingRequest;
    type Output = EmbeddingResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();

        let encodings = self.tokenizer.encode_text(request.inputs)?;

        let transform = EmbeddingTransform::new(self.transform_str())?;

        let results = inference::embedding::embedding(self.session.lock(), &transform, encodings)?;

        Ok(EmbeddingResponse {
            results,
            model_id: self.config.name.clone(),
            metadata: request.metadata,
        })
    }
}
