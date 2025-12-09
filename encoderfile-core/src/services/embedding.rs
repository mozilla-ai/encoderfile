use crate::{
    common::{EmbeddingRequest, EmbeddingResponse, model_type},
    error::ApiError,
    inference,
    runtime::{AppState, InferenceState},
    transforms::EmbeddingTransform,
};

use super::inference::Inference;

impl Inference for AppState<model_type::Embedding> {
    type Input = EmbeddingRequest;
    type Output = EmbeddingResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();

        let session = self.session();

        let encodings = crate::runtime::encode_text(&self.tokenizer(), request.inputs)?;

        let transform = EmbeddingTransform::new(self.transform_str())?;

        let results = inference::embedding::embedding(session, &transform, encodings)?;

        Ok(EmbeddingResponse {
            results,
            model_id: self.model_id.clone(),
            metadata: request.metadata,
        })
    }
}
