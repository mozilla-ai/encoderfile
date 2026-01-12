use crate::common;

tonic::include_proto!("encoderfile.token_classification");

impl From<TokenClassificationRequest> for common::TokenClassificationRequest {
    fn from(val: TokenClassificationRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: Some(val.metadata),
        }
    }
}

impl From<common::TokenClassificationResponse> for TokenClassificationResponse {
    fn from(val: common::TokenClassificationResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|i| i.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata.unwrap_or_default(),
        }
    }
}

impl From<common::TokenClassificationResult> for TokenClassificationResult {
    fn from(val: common::TokenClassificationResult) -> Self {
        Self {
            tokens: val.tokens.into_iter().map(|i| i.into()).collect(),
        }
    }
}

impl From<common::TokenClassification> for TokenClassification {
    fn from(val: common::TokenClassification) -> Self {
        Self {
            token_info: Some(val.token_info.into()),
            scores: val.scores,
            label: val.label,
            score: val.score,
        }
    }
}
