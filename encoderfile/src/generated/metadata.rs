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

impl From<common::model_type::ModelType> for ModelType {
    fn from(val: common::model_type::ModelType) -> Self {
        match val {
            common::model_type::ModelType::Embedding => Self::Embedding,
            common::model_type::ModelType::SequenceClassification => Self::SequenceClassification,
            common::model_type::ModelType::TokenClassification => Self::TokenClassification,
            common::model_type::ModelType::SentenceEmbedding => Self::SentenceEmbedding,
            common::model_type::ModelType::ImageClassification => Self::ImageClassification,
        }
    }
}

impl From<ModelType> for common::model_type::ModelType {
    fn from(val: ModelType) -> Self {
        match val {
            ModelType::Embedding => common::model_type::ModelType::Embedding,
            ModelType::SequenceClassification => common::model_type::ModelType::SequenceClassification,
            ModelType::TokenClassification => common::model_type::ModelType::TokenClassification,
            ModelType::SentenceEmbedding => common::model_type::ModelType::SentenceEmbedding,
            ModelType::ImageClassification => common::model_type::ModelType::ImageClassification,
            ModelType::Unspecified => {
                unreachable!("Unspecified model type. This should not happen.")
            }
        }
    }
}
