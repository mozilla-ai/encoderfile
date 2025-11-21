use crate::common;

tonic::include_proto!("encoderfile.sequence_classification");

impl From<SequenceClassificationRequest>
    for common::SequenceClassificationRequest
{
    fn from(val: SequenceClassificationRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: Some(val.metadata),
        }
    }
}

impl From<common::SequenceClassificationResponse>
    for SequenceClassificationResponse
{
    fn from(val: common::SequenceClassificationResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|i| i.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata.unwrap_or_default(),
        }
    }
}

impl From<common::SequenceClassificationResult>
    for SequenceClassificationResult
{
    fn from(val: common::SequenceClassificationResult) -> Self {
        Self {
            logits: val.logits,
            scores: val.scores,
            predicted_index: val.predicted_index,
            predicted_label: val.predicted_label,
        }
    }
}
