use crate::common;

tonic::include_proto!("encoderfile.sentence_embedding");

impl From<SentenceEmbeddingRequest>
    for common::SentenceEmbeddingRequest
{
    fn from(val: SentenceEmbeddingRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: Some(val.metadata),
        }
    }
}

impl From<common::SentenceEmbeddingResponse>
    for SentenceEmbeddingResponse
{
    fn from(val: common::SentenceEmbeddingResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|i| i.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata.unwrap_or_default(),
        }
    }
}

impl From<common::SentenceEmbedding> for SentenceEmbedding {
    fn from(val: common::SentenceEmbedding) -> Self {
        Self {
            embedding: val.embedding,
        }
    }
}
