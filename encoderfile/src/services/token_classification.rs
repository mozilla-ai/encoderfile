use crate::{
    common::{TokenClassificationRequest, TokenClassificationResponse},
    error::ApiError,
    inference,
    runtime::AppState,
};

#[tracing::instrument(skip_all)]
pub fn token_classification(
    request: impl Into<TokenClassificationRequest>,
    state: &AppState,
) -> Result<TokenClassificationResponse, ApiError> {
    let request = request.into();
    let session = state.session.lock();

    let encodings = crate::runtime::encode_text(&state.tokenizer, request.inputs)?;

    let results = inference::token_classification::token_classification(session, state, encodings)?;

    Ok(TokenClassificationResponse {
        results,
        model_id: state.model_id.clone(),
        metadata: request.metadata,
    })
}
