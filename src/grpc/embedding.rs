use tonic::{Request, Response, Status};

use crate::generated::{
    embedding::{EmbeddingRequest, EmbeddingResponse},
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
        Ok(Response::new(
            crate::services::embedding(request.into_inner())
                .map_err(|e| e.to_tonic_status())?
                .into(),
        ))
    }
}
