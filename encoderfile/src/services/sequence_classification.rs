use crate::{
    common::{SequenceClassificationRequest, SequenceClassificationResponse, model_type},
    error::ApiError,
    inference,
    runtime::AppState,
    transforms::SequenceClassificationTransform,
};

use super::inference::Inference;

impl Inference for AppState<model_type::SequenceClassification> {
    type Input = SequenceClassificationRequest;
    type Output = SequenceClassificationResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();

        let encodings = self.tokenizer.encode_text(request.inputs)?;

        let transform = SequenceClassificationTransform::new(self.transform_str())?;

        let results = inference::sequence_classification::sequence_classification(
            self.session.lock(),
            &transform,
            &self.model_config,
            encodings,
        )?;

        Ok(SequenceClassificationResponse {
            results,
            model_id: self.config.name.clone(),
            metadata: request.metadata,
        })
    }
}
