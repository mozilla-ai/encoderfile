#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Embedding,
    SequenceClassification,
    TokenClassification,
    SentenceEmbedding,
}
