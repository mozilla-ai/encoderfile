use crate::{
    common::{SequenceClassificationRequest, SequenceClassificationResponse, model_type},
    error::ApiError,
    inference,
    runtime::{AppState, InferenceState},
    transforms::SequenceClassificationTransform,
};

use super::inference::Inference;

impl Inference for AppState<model_type::SequenceClassification> {
    type Input = SequenceClassificationRequest;
    type Output = SequenceClassificationResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();

        let session = self.session();

        let encodings = crate::runtime::encode_text(self.tokenizer(), request.inputs)?;

        let transform = SequenceClassificationTransform::new(self.transform_str())?;

        let results = inference::sequence_classification::sequence_classification(
            session,
            &transform,
            &self.config,
            encodings,
        )?;

        Ok(SequenceClassificationResponse {
            results,
            model_id: self.model_id.clone(),
            metadata: request.metadata,
        })
    }
}
