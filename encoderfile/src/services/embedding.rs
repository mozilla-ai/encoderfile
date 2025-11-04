use crate::{
    common::{EmbeddingRequest, EmbeddingResponse},
    error::ApiError,
    inference,
    state::AppState,
};

pub fn embedding(
    request: impl Into<EmbeddingRequest>,
    state: &AppState,
) -> Result<EmbeddingResponse, ApiError> {
    let request = request.into();

    let session = state.session.lock();

    let encodings = crate::tokenizer::encode_text(&state.tokenizer, request.inputs)?;

    let results =
        inference::embedding::embedding(session, &state.config, encodings, request.normalize)?;

    Ok(EmbeddingResponse {
        results,
        model_id: state.model_id.clone(),
        metadata: request.metadata,
    })
}
