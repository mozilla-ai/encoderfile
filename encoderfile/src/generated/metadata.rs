use crate::common;

tonic::include_proto!("encoderfile.metadata");

impl From<common::GetModelMetadataResponse> for GetModelMetadataResponse {
    fn from(val: common::GetModelMetadataResponse) -> Self {
        Self {
            model_id: val.model_id,
            model_type: ModelType::from(val.model_type).into(),
            id2label: val.id2label.unwrap_or_default(),
        }
    }
}

impl From<common::ModelType> for ModelType {
    fn from(val: common::ModelType) -> Self {
        match val {
            common::ModelType::Embedding => Self::Embedding,
            common::ModelType::SequenceClassification => Self::SequenceClassification,
            common::ModelType::TokenClassification => Self::TokenClassification,
            common::ModelType::SentenceEmbedding => Self::SentenceEmbedding,
        }
    }
}

impl From<ModelType> for common::ModelType {
    fn from(val: ModelType) -> Self {
        match val {
            ModelType::Embedding => common::ModelType::Embedding,
            ModelType::SequenceClassification => common::ModelType::SequenceClassification,
            ModelType::TokenClassification => common::ModelType::TokenClassification,
            ModelType::SentenceEmbedding => common::ModelType::SentenceEmbedding,
            ModelType::Unspecified => {
                unreachable!("Unspecified model type. This should not happen.")
            }
        }
    }
}
