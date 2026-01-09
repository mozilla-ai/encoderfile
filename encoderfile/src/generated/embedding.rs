use crate::common;

tonic::include_proto!("encoderfile.embedding");

impl From<EmbeddingRequest> for common::EmbeddingRequest {
    fn from(val: EmbeddingRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: Some(val.metadata),
        }
    }
}

impl From<common::EmbeddingResponse> for EmbeddingResponse {
    fn from(val: common::EmbeddingResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|embs| embs.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata.unwrap_or_default(),
        }
    }
}

impl From<common::TokenEmbeddingSequence> for TokenEmbeddingSequence {
    fn from(val: common::TokenEmbeddingSequence) -> Self {
        Self {
            embeddings: val.embeddings.into_iter().map(|i| i.into()).collect(),
        }
    }
}

impl From<common::TokenEmbedding> for TokenEmbedding {
    fn from(val: common::TokenEmbedding) -> Self {
        TokenEmbedding {
            embedding: val.embedding,
            token_info: val.token_info.map(|i| i.into()),
        }
    }
}
