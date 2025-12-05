#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ModelType {
    Embedding = 1,
    SequenceClassification = 2,
    TokenClassification = 3,
    SentenceEmbedding = 4,
}
