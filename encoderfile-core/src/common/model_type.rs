#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Embedding,
    SequenceClassification,
    TokenClassification,
    SentenceEmbedding,
}

impl From<ModelType> for crate::generated::metadata::ModelType {
    fn from(val: ModelType) -> Self {
        match val {
            ModelType::Embedding => Self::Embedding,
            ModelType::SequenceClassification => Self::SequenceClassification,
            ModelType::TokenClassification => Self::TokenClassification,
            ModelType::SentenceEmbedding => Self::SentenceEmbedding,
        }
    }
}
