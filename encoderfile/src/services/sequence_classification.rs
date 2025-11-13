use crate::{
    common::{SequenceClassificationRequest, SequenceClassificationResponse},
    error::ApiError,
    inference,
    runtime::AppState,
};

#[tracing::instrument(skip_all)]
pub fn sequence_classification(
    request: impl Into<SequenceClassificationRequest>,
    state: &AppState,
) -> Result<SequenceClassificationResponse, ApiError> {
    let request = request.into();
    let session = state.session.lock();

    let encodings = crate::runtime::encode_text(&state.tokenizer, request.inputs)?;

    let results =
        inference::sequence_classification::sequence_classification(session, &state, encodings)?;

    Ok(SequenceClassificationResponse {
        results,
        model_id: state.model_id.clone(),
        metadata: request.metadata,
    })
}
