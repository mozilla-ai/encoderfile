use crate::{
    common::{EmbeddingRequest, EmbeddingResponse, model_type},
    error::ApiError,
    inference,
    runtime::AppState,
};

#[tracing::instrument(skip_all)]
pub fn embedding(
    request: impl Into<EmbeddingRequest>,
    state: &AppState<model_type::Embedding>,
) -> Result<EmbeddingResponse, ApiError> {
    let request = request.into();

    let session = state.session.lock();

    let encodings = crate::runtime::encode_text(&state.tokenizer, request.inputs)?;

    let results = inference::embedding::embedding(session, state, encodings)?;

    Ok(EmbeddingResponse {
        results,
        model_id: state.model_id.clone(),
        metadata: request.metadata,
    })
}
