use crate::config::{ModelType, get_model_type};

pub mod embedding;
pub mod sequence_classification;
pub mod token_classification;

pub fn router() -> axum::Router {
    let builder = tonic::service::Routes::builder().routes();

    match get_model_type() {
        ModelType::Embedding => builder.add_service(self::embedding::embedding_server()),
        ModelType::SequenceClassification => {
            builder.add_service(self::sequence_classification::sequence_classification_server())
        }
        ModelType::TokenClassification => {
            builder.add_service(self::token_classification::token_classification_server())
        }
    }
    .into_axum_router()
}
