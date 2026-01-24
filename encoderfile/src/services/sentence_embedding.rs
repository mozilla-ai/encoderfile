use crate::{
    common::{SentenceEmbeddingRequest, SentenceEmbeddingResponse, model_type},
    error::ApiError,
    inference,
    runtime::AppState,
    transforms::SentenceEmbeddingTransform,
};

use super::inference::Inference;

impl Inference for AppState<model_type::SentenceEmbedding> {
    type Input = SentenceEmbeddingRequest;
    type Output = SentenceEmbeddingResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();

        let encodings = self.tokenizer.encode_text(request.inputs)?;

        let transform = SentenceEmbeddingTransform::new(self.transform_str())?;

        let results = inference::sentence_embedding::sentence_embedding(
            self.session.lock(),
            &transform,
            encodings,
        )?;

        Ok(SentenceEmbeddingResponse {
            results,
            model_id: self.config.name.clone(),
            metadata: request.metadata,
        })
    }
}
