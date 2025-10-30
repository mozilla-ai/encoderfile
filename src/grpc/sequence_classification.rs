use tonic::{Request, Response, Status};

use crate::generated::{
    encoderfile::sequence_classification_server::{
        SequenceClassification, SequenceClassificationServer,
    },
    sequence_classification::{
        SequenceClassificationRequest, SequenceClassificationResponse, SequenceClassificationResult,
    },
};

pub fn sequence_classification_server() -> SequenceClassificationServer<SequenceClassificationService> {
    SequenceClassificationServer::new(SequenceClassificationService)
}

#[derive(Debug, Default)]
pub struct SequenceClassificationService;

#[tonic::async_trait]
impl SequenceClassification for SequenceClassificationService {
    async fn predict(
        &self,
        request: Request<SequenceClassificationRequest>,
    ) -> Result<Response<SequenceClassificationResponse>, Status> {
        let request = request.into_inner();

        if request.inputs.len() == 0 {
            return Err(Status::invalid_argument("Inputs are empty"));
        }

        let classifications: Vec<SequenceClassificationResult> =
            crate::services::sequence_classification(request.inputs)
                .map_err(|e| e.to_tonic_status())?
                .into_iter()
                .map(|i| i.into())
                .collect();

        Ok(Response::new(SequenceClassificationResponse {
            results: classifications,
            metadata: request.metadata,
        }))
    }
}
