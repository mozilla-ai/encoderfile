use crate::{
    common::{SentenceEmbeddingRequest, SentenceEmbeddingResponse},
    error::ApiError,
    inference,
    runtime::AppState,
};

#[tracing::instrument(skip_all)]
pub fn sentence_embedding(
    request: impl Into<SentenceEmbeddingRequest>,
    state: &AppState,
) -> Result<SentenceEmbeddingResponse, ApiError> {
    let request = request.into();

    let session = state.session.lock();

    let encodings = crate::runtime::encode_text(&state.tokenizer, request.inputs)?;

    let results = inference::sentence_embedding::sentence_embedding(session, state, encodings)?;

    Ok(SentenceEmbeddingResponse {
        results,
        model_id: state.model_id.clone(),
        metadata: request.metadata,
    })
}
