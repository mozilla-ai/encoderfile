use std::path::PathBuf;

use encoderfile::model::ModelTypeExt as _;
use encoderfile_core::common::ModelType;

#[test]
pub fn test_embedding() {
    let path = PathBuf::from("../models/embedding/model.onnx");

    assert!(ModelType::Embedding.validate_model(&path).is_ok());
    assert!(
        ModelType::SequenceClassification
            .validate_model(&path)
            .is_err()
    );
    assert!(
        ModelType::TokenClassification
            .validate_model(&path)
            .is_err()
    );
    assert!(ModelType::SentenceEmbedding.validate_model(&path).is_ok())
}

#[test]
pub fn test_sequence_classification() {
    let path = PathBuf::from("../models/sequence_classification/model.onnx");

    assert!(ModelType::Embedding.validate_model(&path).is_err());
    assert!(
        ModelType::SequenceClassification
            .validate_model(&path)
            .is_ok()
    );
    assert!(
        ModelType::TokenClassification
            .validate_model(&path)
            .is_err()
    );
    assert!(ModelType::SentenceEmbedding.validate_model(&path).is_err())
}

#[test]
pub fn test_token_classification() {
    let path = PathBuf::from("../models/token_classification/model.onnx");

    assert!(ModelType::Embedding.validate_model(&path).is_err());
    assert!(
        ModelType::SequenceClassification
            .validate_model(&path)
            .is_err()
    );
    assert!(ModelType::TokenClassification.validate_model(&path).is_ok());
    assert!(ModelType::SentenceEmbedding.validate_model(&path).is_err())
}
