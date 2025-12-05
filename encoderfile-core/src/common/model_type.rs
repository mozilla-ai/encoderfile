#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ModelType {
    Embedding = 0,
    SequenceClassification = 1,
    TokenClassification = 2,
    SentenceEmbedding = 3,
}
