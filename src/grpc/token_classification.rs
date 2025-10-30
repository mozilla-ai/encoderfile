use tonic::{Request, Response, Status};

use crate::generated::{
    encoderfile::token_classification_server::{TokenClassification, TokenClassificationServer},
    token_classification::{
        TokenClassificationRequest, TokenClassificationResponse, TokenClassificationResult,
    },
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
        let request = request.into_inner();

        let classifications = crate::services::token_classification(request.inputs)
            .map_err(|e| e.to_tonic_status())?;

        Ok(Response::new(TokenClassificationResponse {
            results: classifications
                .into_iter()
                .map(|preds| TokenClassificationResult {
                    tokens: preds.into_iter().map(|i| i.into()).collect(),
                })
                .collect(),
            metadata: request.metadata,
        }))
    }
}
