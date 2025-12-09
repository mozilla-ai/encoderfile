use crate::{
    common::{TokenClassificationRequest, TokenClassificationResponse, model_type},
    error::ApiError,
    inference,
    runtime::AppState,
    transforms::TokenClassificationTransform,
};

use super::inference::Inference;

impl Inference for AppState<model_type::TokenClassification> {
    type Input = TokenClassificationRequest;
    type Output = TokenClassificationResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();

        let session = self.session.lock();

        let encodings = crate::runtime::encode_text(&self.tokenizer, request.inputs)?;

        let transform = TokenClassificationTransform::new(self.transform_str())?;

        let results = inference::token_classification::token_classification(
            session,
            &transform,
            &self.config,
            encodings,
        )?;

        Ok(TokenClassificationResponse {
            results,
            model_id: self.model_id.clone(),
            metadata: request.metadata,
        })
    }
}

// #[tracing::instrument(skip_all)]
// pub fn token_classification(
//     request: impl Into<TokenClassificationRequest>,
//     state: &AppState<model_type::TokenClassification>,
// ) -> Result<TokenClassificationResponse, ApiError> {
//     let request = request.into();
//     let session = state.session.lock();

//     let encodings = crate::runtime::encode_text(&state.tokenizer, request.inputs)?;

//     let results = inference::token_classification::token_classification(session, state, encodings)?;

//     Ok(TokenClassificationResponse {
//         results,
//         model_id: state.model_id.clone(),
//         metadata: request.metadata,
//     })
// }
