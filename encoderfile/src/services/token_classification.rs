use crate::{
    common::{TokenClassificationRequest, TokenClassificationResponse},
    error::ApiError,
    inference,
    state::AppState,
};

pub fn token_classification(
    request: impl Into<TokenClassificationRequest>,
    state: &AppState,
) -> Result<TokenClassificationResponse, ApiError> {
    let request = request.into();
    let session = state.session.lock();

    let encodings = crate::model::tokenizer::encode_text(&state.tokenizer, request.inputs)?;

    let results =
        inference::token_classification::token_classification(session, &state.config, encodings)?;

    Ok(TokenClassificationResponse {
        results,
        model_id: state.model_id.clone(),
        metadata: request.metadata,
    })
}
