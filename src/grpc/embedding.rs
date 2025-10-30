use tonic::{Request, Response, Status};

use crate::generated::{
    embedding::{EmbeddingRequest, EmbeddingResponse, TokenEmbeddingSequence},
    encoderfile::embedding_server::{Embedding, EmbeddingServer},
};

pub fn embedding_server() -> EmbeddingServer<EmbeddingService> {
    EmbeddingServer::new(EmbeddingService)
}

#[derive(Debug, Default)]
pub struct EmbeddingService;

#[tonic::async_trait]
impl Embedding for EmbeddingService {
    async fn predict(
        &self,
        request: Request<EmbeddingRequest>,
    ) -> Result<Response<EmbeddingResponse>, Status> {
        let request = request.into_inner();

        if request.inputs.len() == 0 {
            return Err(Status::invalid_argument("Inputs are empty"));
        }

        let embeddings = crate::services::embedding(request.inputs, request.normalize)
            .map_err(|e| e.to_tonic_status())?;

        Ok(Response::new(EmbeddingResponse {
            results: embeddings
                .into_iter()
                .map(|embs| TokenEmbeddingSequence {
                    embeddings: embs.into_iter().map(|i| i.into()).collect(),
                })
                .collect(),
            dim: crate::config::get_model_config().hidden_size,
            metadata: request.metadata,
        }))
    }
}
