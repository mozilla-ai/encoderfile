use tonic::{Request, Response, Status};

use crate::generated::{
    encoderfile::token_classification_server::{TokenClassification, TokenClassificationServer},
    token_classification::{TokenClassificationRequest, TokenClassificationResponse},
};

pub fn token_classification_server() -> TokenClassificationServer<TokenClassificationService> {
    TokenClassificationServer::new(TokenClassificationService)
}

#[derive(Debug, Default)]
pub struct TokenClassificationService;

#[tonic::async_trait]
impl TokenClassification for TokenClassificationService {
    async fn predict(
        &self,
        request: Request<TokenClassificationRequest>,
    ) -> Result<Response<TokenClassificationResponse>, Status> {
        Ok(Response::new(
            crate::services::token_classification(request.into_inner())
                .map_err(|e| e.to_tonic_status())?
                .into(),
        ))
    }
}
