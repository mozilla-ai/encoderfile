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

        let encodings = self.tokenizer.encode_text(request.inputs)?;

        let transform = TokenClassificationTransform::new(self.transform_str())?;

        let results = inference::token_classification::token_classification(
            session,
            &transform,
            &self.model_config,
            encodings,
        )?;

        Ok(TokenClassificationResponse {
            results,
            model_id: self.config.name.clone(),
            metadata: request.metadata,
        })
    }
}
