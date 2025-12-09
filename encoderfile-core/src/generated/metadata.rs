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

impl From<common::ModelTypeEnum> for ModelType {
    fn from(val: common::ModelTypeEnum) -> Self {
        match val {
            common::ModelTypeEnum::Embedding => Self::Embedding,
            common::ModelTypeEnum::SequenceClassification => Self::SequenceClassification,
            common::ModelTypeEnum::TokenClassification => Self::TokenClassification,
            common::ModelTypeEnum::SentenceEmbedding => Self::SentenceEmbedding,
        }
    }
}
